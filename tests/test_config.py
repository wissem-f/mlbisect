from mlbisect.config import Configuration


def test_configuration_id_is_stable() -> None:
    first = Configuration.from_axes(["prompt", "data"])
    second = Configuration.from_axes(["data", "prompt"])

    assert first == second
    assert first.id == "{data,prompt}"
    assert Configuration().id == "{}"
    assert first.ordered(["data", "deps", "prompt"]) == ("data", "prompt")
