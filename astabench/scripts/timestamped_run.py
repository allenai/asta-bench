import datetime
import fcntl
import json
import os
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    stdout_file: str
    stderr_file: str
    timed_out: bool = False
    inactivity_timed_out: bool = False
    exit_code: int | None = None


@dataclass
class ExecCmd:
    cmd: list[str]

    skip: bool

    name_prefix: str | None = None
    """optionally, prepend this name to the log file"""


def mktimestamp():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def get_terminal_size():
    """Get current terminal dimensions"""
    try:
        h, w = struct.unpack(
            "HH", fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0\0\0\0")
        )
        return h, w
    except:
        return 24, 80  # Default fallback


def subst_log_dir(cmd: list[str], log_dir: str) -> list[str]:
    newlst = cmd.copy()
    for idx, item in enumerate(newlst):
        if item.startswith("--log-dir="):
            newlst[idx] = f"--log-dir={log_dir}"
            return newlst
        elif item == "--log-dir":
            newlst[idx + 1] = log_dir
            return newlst
    raise ValueError("No --log-dir in cmd")


def run_exp_cmds(exp_cmds: list[list[str] | ExecCmd]):
    if subprocess.run(["which", "unbuffer"]).returncode != 0:
        print("Installing 'unbuffer' which is needed for timestamping")
        subprocess.run(["apt-get", "install", "-y", "expect"])

    exp_cmds: list[ExecCmd] = [
        (x if isinstance(x, ExecCmd) else ExecCmd(cmd=x, skip=False)) for x in exp_cmds
    ]

    print("Preparing to run the following commands:")
    for idx, cmd in enumerate(exp_cmds):
        namestr = "" if not cmd.name_prefix else f" ({cmd.name_prefix})"
        skipstr = "(SKIP) " if cmd.skip else ""
        print(f"{idx+1}{namestr}. {skipstr}{' '.join(cmd.cmd)}")
    print()

    total_runs = len(exp_cmds)

    successful_runs = 0
    failed_argsets = []
    successful_argsets = []

    for i, cmd_obj in enumerate(exp_cmds):
        print(f"\n--- Run {i+1}/{total_runs} ---")
        if cmd_obj.skip:
            print("Skipping this run...")
            continue

        cmd = cmd_obj.cmd

        datestr = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
        exp_dir = (
            Path("./logs")
            / f"{cmd_obj.name_prefix or ''}asta-bench_exp_r{i+1}_{datestr}"
        )
        print(f"storing results in {exp_dir}")

        cmd = subst_log_dir(cmd, exp_dir)

        exec_result = run_cmd_with_capture(
            cmd,
            timeout=60 * 60 * 32,
            inactivity_timeout=60 * 30,
            log_dir=exp_dir,
        )

        if exec_result.exit_code == 0:
            successful_runs += 1
            successful_argsets.append((cmd, exec_result))
        else:
            failed_argsets.append((cmd, exec_result))
            # print("Warning: run failed but not cleaning up sandboxes")
            print("Cleaning up sandboxes...")
            subprocess.run(["uv", "run", "inspect", "sandbox", "cleanup", "docker"])
            print("Cleaned up sandboxes.")

        print(
            f"Completed {i+1}/{total_runs} runs. Success: {successful_runs}, Failed: {len(failed_argsets)}"
        )

    print(
        f"\nAll experiments completed. Success: {successful_runs}, Failed: {len(failed_argsets)}"
    )
    print("Successful argsets:")
    for argset, exec_result in successful_argsets:
        print(f"    - {' '.join(argset)}")
        print(f"        - Exec result: {exec_result}")
    print()
    print("Failed argsets:")
    for argset, exec_result in failed_argsets:
        print(f"    - {' '.join(argset)}")
        print(f"        - Exec result: {exec_result}")


def add_timestamp_to_output(
    pipe, log_file: str, to_stderr: bool = False, last_activity_time=None
):
    """Read from pipe, add timestamps, log to file, and optionally print."""
    break_reason = "loop"
    for line in iter(pipe.readline, ""):
        if not line:
            break_reason = "empty_read"
            break
        try:
            timestamp = mktimestamp()
            timestamped_line = f"{timestamp} {line}"

            log_file.write(timestamped_line)
            log_file.flush()

            print(
                timestamped_line, end="", file=(sys.stderr if to_stderr else sys.stdout)
            )

            # Update last activity time if provided
            if last_activity_time is not None:
                last_activity_time[0] = time.perf_counter()
        except Exception as e:
            break_reason = f"error: {e}"
            break

    msg = f"Breaking add_timestamp loop at {datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} for reason: {break_reason}\n"
    print(msg, end="")
    with open("runner_meta.log", "a") as f:
        f.write(msg)


def terminate_process_gracefully(process: subprocess.Popen, grace_period: int = 20):
    """
    Attempts to terminate a process gracefully with SIGTERM before
    resorting to SIGKILL.

    Args:
        process: subprocess.Popen instance
        grace_period: seconds to wait after SIGTERM before sending SIGKILL

    Returns:
        exit_code of the process
    """
    if process.poll() is not None:
        return process.returncode

    def grace_period_sigint_handler(sig, frame):
        print("\nInterrupt received during graceful shutdown, killing immediately...")
        process.kill()

    # Set temporary handler for grace period
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, grace_period_sigint_handler)

    try:
        print("Sending SIGTERM, waiting for graceful exit...")
        process.terminate()
        process.wait(timeout=grace_period)

        if process.poll() is None:
            print("Process did not terminate gracefully, sending SIGKILL...")
            process.kill()
            return process.wait(timeout=5) or -1

        return process.returncode
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint)


def run_cmd_with_capture(
    cmd: list[str],
    timeout: int | None = None,
    inactivity_timeout: int | None = None,
    log_dir: str | Path = "./logs/",
) -> int:
    print(f"Running command: {' '.join(cmd)}")

    datestr = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    log_dir = Path(log_dir)

    os.makedirs(log_dir, exist_ok=True)
    stdout_file = (log_dir / f"rawlogs_{datestr}_stdout.log").as_posix()
    stderr_file = (log_dir / f"rawlogs_{datestr}_stderr.log").as_posix()

    # Check if log files already exist
    if os.path.exists(stderr_file):
        print("Error: Log file already exists; not overwriting. Abort.")
        return 1

    # Some processes read the terminal size and output lines of that length
    # (e.g. to draw horizontal separators), so we need to set a fake terminal
    # size to account for the timestamp width
    timestamp_width = len(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] "))

    rows, cols = get_terminal_size()
    adjusted_cols = max(
        20, cols - timestamp_width - 2
    )  # Adjust width, leave some margin

    env = os.environ.copy()
    env["COLUMNS"] = str(adjusted_cols)
    print(f"Setting adjusted width to {adjusted_cols}")

    # Variables for tracking process state
    process = None
    exit_code = 1
    timed_out = False
    inactivity_timed_out = False

    exec_result = ExecResult(
        exit_code=None,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        timed_out=False,
        inactivity_timed_out=False,
    )

    # Shared variable to track last activity time (using list for thread-safe updates)
    last_activity_time = [time.perf_counter()] if inactivity_timeout else None

    # Setup signal handler for Ctrl-C (SIGINT)
    def sigint_handler(sig, frame):
        nonlocal process
        print("Interrupting process with SIGINT")
        process.send_signal(signal.SIGINT)

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    start_time = time.perf_counter()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Line-buffer
            bufsize=1,
            env=env,
            # Run in a new session for signal handling
            start_new_session=True,
        )

        with open(stdout_file, "w") as stdout_log, open(stderr_file, "w") as stderr_log:
            stderr_log.write(
                f"(runner) {mktimestamp()} Running command: {json.dumps(cmd)}\n"
            )

            # Create threads to handle stdout and stderr
            stdout_thread = threading.Thread(
                target=add_timestamp_to_output,
                args=(process.stdout, stdout_log, False, last_activity_time),
                daemon=True,
            )

            stderr_thread = threading.Thread(
                target=add_timestamp_to_output,
                args=(process.stderr, stderr_log, True, last_activity_time),
                daemon=True,
            )

            stdout_thread.start()
            stderr_thread.start()

            # Smart waiting logic for timeouts
            try:
                while process.poll() is None:
                    current_time = time.perf_counter()

                    # Calculate when timeouts would occur
                    remaining_timeout = None
                    if timeout:
                        remaining_timeout = timeout - (current_time - start_time)
                        if remaining_timeout <= 0:
                            timed_out = True
                            print(f"Process timed out after {timeout} seconds")
                            exit_code = terminate_process_gracefully(process)
                            break

                    remaining_inactivity_timeout = None
                    if inactivity_timeout and last_activity_time:
                        remaining_inactivity_timeout = inactivity_timeout - (
                            current_time - last_activity_time[0]
                        )
                        if remaining_inactivity_timeout <= 0:
                            inactivity_timed_out = True
                            print(
                                f"Process timed out due to inactivity after {inactivity_timeout} seconds (total runtime: {(current_time - start_time):0.2f} seconds)"
                            )
                            exit_code = terminate_process_gracefully(process)
                            break

                    # Wait until the sooner of the two timeouts (or just one if only one is set)
                    wait_timeout = None
                    if (
                        remaining_timeout is not None
                        and remaining_inactivity_timeout is not None
                    ):
                        wait_timeout = min(
                            remaining_timeout, remaining_inactivity_timeout
                        )
                    elif remaining_timeout is not None:
                        wait_timeout = remaining_timeout
                    elif remaining_inactivity_timeout is not None:
                        wait_timeout = remaining_inactivity_timeout

                    # Wait for process to finish or timeout
                    try:
                        exit_code = process.wait(timeout=wait_timeout)
                        break  # Process finished normally
                    except subprocess.TimeoutExpired:
                        # Continue the loop to check which timeout occurred
                        continue

                # If we exit the loop without timeout, get the return code
                if (
                    process.poll() is not None
                    and not timed_out
                    and not inactivity_timed_out
                ):
                    exit_code = process.returncode

            except subprocess.TimeoutExpired:
                # This shouldn't happen with our new logic, but keep as fallback
                timed_out = True
                print(f"Process timed out after {timeout} seconds")
                exit_code = terminate_process_gracefully(process)

            # Give output threads a chance to finish
            time.sleep(0.5)

            # Force close pipes to ensure threads don't hang
            process.stdout.close()
            process.stderr.close()

            elapsed_time = time.perf_counter() - start_time

            timeout_msg = ""
            if timed_out:
                timeout_msg = " (TIMED OUT)"
            elif inactivity_timed_out:
                timeout_msg = " (INACTIVITY TIMEOUT)"

            completion_msg = f"(runner) {mktimestamp()} Finished run {datestr} with code {exit_code}{timeout_msg} (elapsed time: {elapsed_time:.2f}s)\n"
            print(completion_msg, end="")
            stderr_log.write(completion_msg)

    except Exception as e:
        print(f"Error in run_cmd_with_capture: {e}")
        exit_code = 1

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint)

        # Ensure process is terminated if still running
        if process and process.poll() is None:
            exit_code = terminate_process_gracefully(process)

    exec_result.exit_code = exit_code
    exec_result.timed_out = timed_out
    exec_result.inactivity_timed_out = inactivity_timed_out
    return exec_result
