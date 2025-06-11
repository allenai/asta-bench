import json

import pytest
from inspect_ai.tool import ToolError

from astabench.tools.table import (
    Cell,
    Column,
    EditColumnValuesParams,
    TableResponsePayload,
    table_editor,
)


@pytest.mark.asyncio
async def test_add_and_get_columns_and_papers():
    editor = table_editor(instance="t1")
    # add columns
    cols = [
        Column(title="C1", description="Desc1"),
        Column(title="C2", description="Desc2"),
    ]
    resp1 = await editor(action="add_columns", columns=cols)
    data1 = json.loads(str(resp1))
    state1 = TableResponsePayload.model_validate(data1).table
    assert [c.title for c in state1.columns] == ["C1", "C2"]
    # add papers
    resp2 = await editor(action="add_papers", paper_ids=["p1", "p2"])
    data2 = json.loads(str(resp2))
    state2 = TableResponsePayload.model_validate(data2).table
    assert state2.paper_ids == ["p1", "p2"]
    # get current state
    resp3 = await editor(action="get")
    data3 = json.loads(str(resp3))
    state3 = TableResponsePayload.model_validate(data3).table
    assert state3 == state2
    assert [c.title for c in state3.columns] == ["C1", "C2"]


@pytest.mark.asyncio
async def test_edit_and_remove_columns_and_papers():
    editor = table_editor(instance="t2")
    # setup
    await editor(action="add_columns", columns=[Column(title="Col", description="D")])
    await editor(action="add_papers", paper_ids=["a", "b"])
    # edit values
    resp1 = await editor(
        action="edit_column_values",
        edit_params=EditColumnValuesParams(
            column="Col",
            updates=[Cell(paper_id="a", value="v1"), Cell(paper_id="b", value="v2")],
        ),
    )
    data1 = json.loads(str(resp1))
    state1 = TableResponsePayload.model_validate(data1).table
    col = state1.columns[0]
    assert {c.paper_id: c.value for c in col.values} == {"a": "v1", "b": "v2"}
    # remove column
    resp2 = await editor(action="remove_columns", remove_columns=["Col"])
    data2 = json.loads(str(resp2))
    state2 = TableResponsePayload.model_validate(data2).table
    assert state2.columns == []
    # remove paper (and its values)
    # re-add to test removal
    await editor(action="add_columns", columns=[Column(title="Col", description="D")])
    await editor(action="add_papers", paper_ids=["x"])
    resp3 = await editor(action="remove_papers", paper_ids=["x"])
    data3 = json.loads(str(resp3))
    state3 = TableResponsePayload.model_validate(data3).table
    assert "x" not in state3.paper_ids
    assert all(
        all(cell.paper_id != "x" for cell in col.values) for col in state3.columns
    )


@pytest.mark.asyncio
async def test_invalid_actions_and_missing_params_raise():
    editor = table_editor(instance="t3")
    with pytest.raises(ToolError):
        await editor(action="unknown")
    # missing required parameters for each action
    with pytest.raises(ToolError):
        await editor(action="add_columns")
    with pytest.raises(ToolError):
        await editor(action="add_papers")
    with pytest.raises(ToolError):
        await editor(action="edit_column_values")
    with pytest.raises(ToolError):
        await editor(action="remove_columns")
    with pytest.raises(ToolError):
        await editor(action="remove_papers")
    # editing a non-existent column
    with pytest.raises(ToolError):
        bad_edit = [
            Cell(paper_id="p", value="v"),
        ]
        await editor(
            action="edit_column_values",
            edit_params=EditColumnValuesParams(column="Nope", updates=bad_edit),
        )


@pytest.mark.asyncio
async def test_add_columns_with_initial_values():
    editor = table_editor(instance="init_vals")
    # pre-add papers so values map can reference existing paper_ids
    await editor(action="add_papers", paper_ids=["p1", "p2"])
    # add column with initial values
    col = Column(
        title="C",
        description="D",
        values=[
            Cell(paper_id="p1", value="x1"),
            Cell(paper_id="p2", value="x2"),
        ],
    )
    resp = await editor(action="add_columns", columns=[col])
    data = json.loads(str(resp))
    state = TableResponsePayload.model_validate(data).table
    assert {c.paper_id: c.value for c in state.columns[0].values} == {
        "p1": "x1",
        "p2": "x2",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action, kwargs",
    [
        ("add_columns", {}),
        ("add_papers", {}),
        ("edit_column_values", {}),
        ("remove_columns", {}),
        ("remove_papers", {}),
    ],
)
async def test_missing_params_raise(action, kwargs):
    editor = table_editor(instance="mp")
    with pytest.raises(ToolError):
        await editor(action=action, **kwargs)


@pytest.mark.asyncio
async def test_undo_reverts_last_action_table():
    editor = table_editor(instance="undo_tbl")
    # initial state: add a paper
    resp1 = await editor(action="add_papers", paper_ids=["p1"])
    data1 = json.loads(str(resp1))
    state1 = TableResponsePayload.model_validate(data1).table
    # perform a second action: add a column
    col = Column(title="C", description="D")
    resp2 = await editor(action="add_columns", columns=[col])
    data2 = json.loads(str(resp2))
    state2 = TableResponsePayload.model_validate(data2).table
    assert [c.title for c in state2.columns] == ["C"]
    # undo the last action (add_columns)
    resp3 = await editor(action="undo")
    data3 = json.loads(str(resp3))
    state3 = TableResponsePayload.model_validate(data3).table
    # state should revert to after first action
    assert state3 == state1


@pytest.mark.asyncio
async def test_add_columns_duplicate_title_error():
    editor = table_editor(instance="dup_col1")
    # add initial column "C"
    await editor(action="add_columns", columns=[Column(title="C", description="D")])
    # adding "C" again should fail
    with pytest.raises(ToolError):
        await editor(
            action="add_columns", columns=[Column(title="C", description="D2")]
        )


# Additional tests


@pytest.mark.asyncio
async def test_insert_before_column():
    editor = table_editor(instance="insert_after")
    # seed with two columns
    await editor(
        action="add_columns",
        columns=[
            Column(title="C1", description="D1"),
            Column(title="C3", description="D3"),
        ],
    )
    # insert C2 before C3
    resp = await editor(
        action="add_columns",
        columns=[Column(title="C2", description="D2")],
        before_column="C3",
    )
    data = json.loads(str(resp))
    state = TableResponsePayload.model_validate(data).table
    assert [c.title for c in state.columns] == ["C1", "C2", "C3"]


@pytest.mark.asyncio
async def test_insert_before_column_invalid():
    editor = table_editor(instance="insert_after_invalid")
    # seed with one column
    await editor(
        action="add_columns",
        columns=[Column(title="Col1", description="D1")],
    )
    # attempt invalid insert
    with pytest.raises(ToolError) as exc:
        await editor(
            action="add_columns",
            columns=[Column(title="Col2", description="D2")],
            before_column="NonExistent",
        )
    assert "Column to insert before not found" in str(exc.value)


@pytest.mark.asyncio
async def test_missing_cells_after_adds():
    editor = table_editor(instance="miss_cells")
    # add papers and one column with value for only one paper
    await editor(action="add_papers", paper_ids=["p1", "p2"])
    await editor(
        action="add_columns",
        columns=[
            Column(
                title="Col",
                description="Desc",
                values=[Cell(paper_id="p1", value="v1")],
            )
        ],
    )
    resp = await editor(action="get")
    data = json.loads(str(resp))
    state = TableResponsePayload.model_validate(data).table
    assert state.missing_cells["Col"] == ["p2"]


@pytest.mark.asyncio
async def test_orphaned_cells_returned_on_remove():
    editor = table_editor(instance="orphan_test")
    # add a paper and a column with a cell
    await editor(action="add_papers", paper_ids=["pX"])
    await editor(
        action="add_columns",
        columns=[
            Column(
                title="Col", description="D", values=[Cell(paper_id="pX", value="v")]
            )
        ],
    )
    raw = await editor(action="remove_papers", paper_ids=["pX"])
    data = json.loads(str(raw))
    assert "orphaned_cells" in data
    assert data["orphaned_cells"][0]["paper_id"] == "pX"


@pytest.mark.asyncio
async def test_undo_not_consumed_by_failed_action():
    editor = table_editor(instance="hist_rollback", history_limit=2)
    # first valid add
    await editor(action="add_columns", columns=[Column(title="A", description="D")])
    # failed add duplicate should not consume history
    with pytest.raises(ToolError):
        await editor(
            action="add_columns", columns=[Column(title="A", description="D2")]
        )
    # undo should revert to empty (history only had the first action)
    resp = await editor(action="undo")
    data = json.loads(str(resp))
    state = TableResponsePayload.model_validate(data).table
    assert state.columns == []
