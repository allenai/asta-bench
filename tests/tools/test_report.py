import json

import pytest
from inspect_ai.tool import ToolError

from astabench.tools.report import (
    EditSectionParams,
    ReportResponsePayload,
    report_editor,
)
from astabench.types.sqa import Citation, SQASection


@pytest.mark.asyncio
async def test_add_and_get_sections():
    editor = report_editor(instance="test1")
    # Bulk-add one section
    resp_json = await editor(
        action="add_sections",
        sections=[SQASection(title="Intro", text="This is intro.", citations=[])],
    )
    data = json.loads(str(resp_json))
    payload = ReportResponsePayload.model_validate(data)
    resp = payload.report
    assert len(resp.sections) == 1
    assert resp.sections[0].title == "Intro"
    assert resp.sections[0].text == "This is intro."

    # Get current state without changes
    resp2_json = await editor(action="get")
    data2 = json.loads(str(resp2_json))
    payload2 = ReportResponsePayload.model_validate(data2)
    resp2 = payload2.report
    assert resp2 == resp


@pytest.mark.asyncio
async def test_edit_and_remove_section():
    editor = report_editor(instance="test2")
    # Initialize with one section
    await editor(
        action="add_sections",
        sections=[SQASection(title="S1", text="T1", citations=[])],
    )
    # Edit section
    resp_json = await editor(
        action="edit_section",
        edit_params=EditSectionParams(
            target_section="S1",
            new_title="S1 Edited",
            old=None,
            text="T1 Edited",
            citations=[],
        ),
    )
    data = json.loads(str(resp_json))
    payload = ReportResponsePayload.model_validate(data)
    resp = payload.report
    assert resp.sections[0].title == "S1 Edited"
    assert resp.sections[0].text == "T1 Edited"
    # Remove section
    resp2_json = await editor(action="remove_sections", remove_sections=["S1 Edited"])
    data2 = json.loads(str(resp2_json))
    payload2 = ReportResponsePayload.model_validate(data2)
    resp2 = payload2.report
    assert len(resp2.sections) == 0


@pytest.mark.asyncio
async def test_clear_resets_state():
    editor = report_editor(instance="test3")
    await editor(
        action="add_sections", sections=[SQASection(title="X", text="Y", citations=[])]
    )
    resp1_json = await editor(action="get")
    data1 = json.loads(str(resp1_json))
    payload1 = ReportResponsePayload.model_validate(data1)
    resp1 = payload1.report
    assert len(resp1.sections) == 1
    # Adding an empty sections list should _not_ clear existing state
    resp_clear_json = await editor(action="add_sections", sections=[])
    data_clear = json.loads(str(resp_clear_json))
    payload_clear = ReportResponsePayload.model_validate(data_clear)
    resp_clear = payload_clear.report
    # state remains unchanged
    assert len(resp_clear.sections) == 1
    assert resp_clear == resp1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action, kwargs",
    [
        ("add_sections", {}),
        ("edit_section", {}),
        ("remove_sections", {}),
    ],
)
async def test_missing_params_raise(action, kwargs):
    editor = report_editor(instance="mp")
    with pytest.raises(ToolError):
        await editor(action=action, **kwargs)


@pytest.mark.asyncio
async def test_invalid_actions_raise():
    editor = report_editor(instance="test5")
    with pytest.raises(ToolError):
        await editor(action="unknown")
    # Citation without section (bulk-add interface) removed
    with pytest.raises(ToolError):
        await editor(action="remove_sections", remove_sections=None)


# Test substring replace emits orphans
@pytest.mark.asyncio
async def test_substring_replace_emits_orphans():
    editor = report_editor(instance="orphans")
    # start with a section containing citation marker [1]
    await editor(
        action="add_sections",
        sections=[
            SQASection(
                title="Sec",
                text="Keep [1] remove",
                citations=[Citation(id="[1]", snippets=["s"], metadata=None)],
            )
        ],
    )
    # replacing a word that doesn't remove [1] should not orphan
    r1 = await editor(
        action="edit_section",
        edit_params=EditSectionParams(
            target_section="Sec",
            new_title="Sec",
            old="remove",
            text="gone",
            citations=[Citation(id="[1]", snippets=["s"], metadata=None)],
        ),
    )
    data_r1 = json.loads(str(r1))
    payload_r1 = ReportResponsePayload.model_validate(data_r1)
    assert payload_r1.orphaned_citations == []
    # now remove the marker via substring replace
    payload = await editor(
        action="edit_section",
        edit_params=EditSectionParams(
            target_section="Sec",
            new_title="Sec",
            old="[1]",
            text="",
            citations=[],
        ),
    )
    data_r2 = json.loads(str(payload))
    payload_r2 = ReportResponsePayload.model_validate(data_r2)
    assert (
        payload_r2.orphaned_citations and payload_r2.orphaned_citations[0].id == "[1]"
    )


@pytest.mark.asyncio
async def test_undo_reverts_last_action():
    editor = report_editor(instance="undo_test")
    # initial state: add one section
    await editor(
        action="add_sections",
        sections=[SQASection(title="Sec", text="Initial", citations=[])],
    )
    # apply an edit
    await editor(
        action="edit_section",
        edit_params=EditSectionParams(
            target_section="Sec",
            new_title="Sec Edited",
            old="",
            text="",
            citations=[],
        ),
    )
    resp2_json = await editor(action="get")
    data2 = json.loads(str(resp2_json))
    payload2 = ReportResponsePayload.model_validate(data2)
    resp2 = payload2.report
    assert resp2.sections[0].title == "Sec Edited"
    # undo the edit
    resp3_json = await editor(action="undo")
    data3 = json.loads(str(resp3_json))
    payload3 = ReportResponsePayload.model_validate(data3)
    resp3 = payload3.report
    # state should match the original section
    assert len(resp3.sections) == 1
    assert resp3.sections[0].title == "Sec"
    assert resp3.sections[0].text == "Initial"


@pytest.mark.asyncio
async def test_add_sections_duplicate_title_error():
    editor = report_editor(instance="dup1")
    # first add section "A"
    await editor(
        action="add_sections", sections=[SQASection(title="A", text="", citations=[])]
    )
    # adding "A" again should fail
    with pytest.raises(ToolError):
        await editor(
            action="add_sections",
            sections=[SQASection(title="A", text="", citations=[])],
        )


@pytest.mark.asyncio
async def test_add_sections_payload_duplicate_titles_error():
    editor = report_editor(instance="dup2")
    # payload contains two sections both titled "B"
    with pytest.raises(ToolError):
        await editor(
            action="add_sections",
            sections=[
                SQASection(title="B", text="", citations=[]),
                SQASection(title="B", text="", citations=[]),
            ],
        )


@pytest.mark.asyncio
async def test_edit_section_duplicate_title_error():
    editor = report_editor(instance="dup3")
    # add two distinct sections
    await editor(
        action="add_sections",
        sections=[
            SQASection(title="X", text="", citations=[]),
            SQASection(title="Y", text="", citations=[]),
        ],
    )
    # attempting to rename "Y" to "X" should fail
    with pytest.raises(ToolError):
        await editor(
            action="edit_section",
            edit_params=EditSectionParams(
                target_section="Y",
                new_title="X",
                old="",
                text="",
                citations=[],
            ),
        )


# Additional tests
@pytest.mark.asyncio
async def test_insert_before_section():
    editor = report_editor(instance="pos_test")
    # seed with two sections
    await editor(
        action="add_sections",
        sections=[
            SQASection(title="A", text="a", citations=[]),
            SQASection(title="C", text="c", citations=[]),
        ],
    )
    # insert B between A and C by inserting before C
    resp_json = await editor(
        action="add_sections",
        sections=[SQASection(title="B", text="b", citations=[])],
        before_section="C",
    )
    data = json.loads(str(resp_json))
    payload = ReportResponsePayload.model_validate(data)
    resp = payload.report
    titles = [s.title for s in resp.sections]
    assert titles == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_undo_boundary():
    editor = report_editor(instance="undo_boundary")
    # perform one mutation to generate history
    await editor(
        action="add_sections",
        sections=[SQASection(title="X", text="x", citations=[])],
    )
    # undo should revert to empty
    resp1_json = await editor(action="undo")
    data1 = json.loads(str(resp1_json))
    payload1 = ReportResponsePayload.model_validate(data1)
    resp1 = payload1.report
    assert resp1.sections == []
    # further undo should error
    with pytest.raises(ToolError):
        await editor(action="undo")


@pytest.mark.asyncio
async def test_history_limit_and_undo():
    editor = report_editor(instance="history_test", history_limit=3)
    # add four sections to overflow history limit
    for title in ["A", "B", "C", "D"]:
        await editor(
            action="add_sections",
            sections=[SQASection(title=title, text=title.lower(), citations=[])],
        )
    # first undo -> back to A, B, C
    resp1_json = await editor(action="undo")
    data1 = json.loads(str(resp1_json))
    payload1 = ReportResponsePayload.model_validate(data1)
    resp1 = payload1.report
    assert [s.title for s in resp1.sections] == ["A", "B", "C"]
    # second undo -> A, B
    resp2_json = await editor(action="undo")
    data2 = json.loads(str(resp2_json))
    payload2 = ReportResponsePayload.model_validate(data2)
    resp2 = payload2.report
    assert [s.title for s in resp2.sections] == ["A", "B"]
    # third undo -> A
    resp3_json = await editor(action="undo")
    data3 = json.loads(str(resp3_json))
    payload3 = ReportResponsePayload.model_validate(data3)
    resp3 = payload3.report
    assert [s.title for s in resp3.sections] == ["A"]
    # fourth undo should error
    with pytest.raises(ToolError):
        await editor(action="undo")


# Additional tests per instructions
@pytest.mark.asyncio
async def test_insert_before_section_invalid():
    editor = report_editor(instance="invalid_insert_test")
    # seed with one section
    await editor(
        action="add_sections",
        sections=[SQASection(title="One", text="one", citations=[])],
    )
    # attempt to insert before a non-existent section
    with pytest.raises(ToolError) as exc:
        await editor(
            action="add_sections",
            sections=[SQASection(title="Two", text="two", citations=[])],
            before_section="NonExistent",
        )
    assert "Section to insert before not found" in str(exc.value)


@pytest.mark.asyncio
async def test_edit_with_no_citations_param_leaves_existing():
    editor = report_editor(instance="citations_preserve")
    # add section with one citation
    cite = Citation(id="[1]", snippets=["s"], metadata=None)
    await editor(
        action="add_sections",
        sections=[SQASection(title="Sec", text="x [1]", citations=[cite])],
    )
    # edit only the text, omit citations param
    resp_json = await editor(
        action="edit_section",
        edit_params=EditSectionParams(
            target_section="Sec",
            new_title=None,
            old=None,
            text="updated [1]",
            citations=None,
        ),
    )
    data = json.loads(str(resp_json))
    payload = ReportResponsePayload.model_validate(data)
    resp = payload.report
    # citation should still be present
    assert resp.sections[0].citations and resp.sections[0].citations[0].id == "[1]"


@pytest.mark.asyncio
async def test_undo_after_failed_mutation():
    editor = report_editor(instance="undo_rollback_test", history_limit=2)
    # initial add
    await editor(
        action="add_sections",
        sections=[SQASection(title="A", text="a", citations=[])],
    )
    # attempt invalid add (duplicate title)
    with pytest.raises(ToolError):
        await editor(
            action="add_sections",
            sections=[SQASection(title="A", text="a2", citations=[])],
        )
    # state should remain as after first add
    get_json = await editor(action="get")
    data_get = json.loads(str(get_json))
    resp = ReportResponsePayload.model_validate(data_get)
    report = resp.report
    assert [s.title for s in report.sections] == ["A"]
    # undo should revert to empty
    await editor(action="undo")
    get2_json = await editor(action="get")
    data_get2 = json.loads(str(get2_json))
    resp2 = ReportResponsePayload.model_validate(data_get2)
    report2 = resp2.report
    assert report2.sections == []
