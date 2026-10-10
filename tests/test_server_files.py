"""Tests that the files a conversation made are shown to the page without running anything they hold."""

import os
from pathlib import Path

import pytest

from virtual_lab.server import files as files_module
from virtual_lab.server.errors import ApiError
from virtual_lab.server.files import (
    CONTENT_SECURITY_POLICY,
    file_response,
    list_files,
    media_type,
    resolve_inside,
)


def write(root: Path, name: str, content: bytes | str = b"data") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())

    return path


class TestResolving:
    def test_a_file_is_found_by_its_path_from_the_root(self, tmp_path: Path) -> None:
        written = write(tmp_path, "figures/plot.png")

        assert resolve_inside(tmp_path, "figures/plot.png") == written.resolve()

    @pytest.mark.parametrize(
        "relative",
        [
            "",
            "/etc/passwd",
            "../outside.txt",
            "figures/../../outside.txt",
            "a\x00b.txt",
            "missing.txt",
        ],
    )
    def test_a_path_that_leaves_the_root_or_leads_nowhere_is_not_a_file(self, tmp_path: Path, relative: str) -> None:
        root = tmp_path / "root"
        write(root, "figures/plot.png")
        write(tmp_path, "outside.txt")

        with pytest.raises(ApiError) as caught:
            resolve_inside(root, relative)

        assert caught.value.status == 404

    def test_a_path_is_from_the_root_even_where_the_root_holds_the_file_it_names(self, tmp_path: Path) -> None:
        written = write(tmp_path, "figures/plot.png")

        with pytest.raises(ApiError):
            resolve_inside(tmp_path, str(written))

    def test_a_path_that_goes_up_and_down_again_is_not_a_way_around_what_is_shown(self, tmp_path: Path) -> None:
        write(tmp_path, "uploads/data.csv")
        write(tmp_path, "lab/notes.txt")

        for relative in ("uploads/../lab/notes.txt", "uploads/../uploads/data.csv"):
            with pytest.raises(ApiError):
                resolve_inside(tmp_path, relative, ("uploads",))

    def test_a_directory_is_not_a_file(self, tmp_path: Path) -> None:
        write(tmp_path, "figures/plot.png")

        with pytest.raises(ApiError):
            resolve_inside(tmp_path, "figures")

    def test_a_link_to_a_file_outside_the_root_is_not_followed(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        secret = write(tmp_path, "secret.txt", "keys")
        (root / "innocent.txt").symlink_to(secret)
        (root / "folder").symlink_to(tmp_path, target_is_directory=True)

        for relative in ("innocent.txt", "folder/secret.txt"):
            with pytest.raises(ApiError):
                resolve_inside(root, relative)

    def test_a_link_to_a_file_inside_the_root_is_shown(self, tmp_path: Path) -> None:
        target = write(tmp_path, "real.txt", "inside")
        (tmp_path / "alias.txt").symlink_to(target)

        assert resolve_inside(tmp_path, "alias.txt").read_text() == "inside"

    def test_only_the_parts_that_are_shown_are_found(self, tmp_path: Path) -> None:
        write(tmp_path, "uploads/data.csv")
        write(tmp_path, "chat.json")
        write(tmp_path, "uploads_extra/data.csv")

        assert resolve_inside(tmp_path, "uploads/data.csv", ("uploads",)).name == "data.csv"
        for relative in ("chat.json", "uploads_extra/data.csv", "uploads", ""):
            with pytest.raises(ApiError):
                resolve_inside(tmp_path, relative, ("uploads",))


class TestListing:
    def test_files_are_listed_by_path_with_their_size_and_type(self, tmp_path: Path) -> None:
        write(tmp_path, "b.txt", "four")
        write(tmp_path, "a/plot.png", b"png")
        write(tmp_path, "a/data.csv", "x,y\n")

        found, truncated = list_files(tmp_path)

        assert not truncated
        assert [item["path"] for item in found] == ["a/data.csv", "a/plot.png", "b.txt"]
        assert [item["size"] for item in found] == [4, 3, 4]
        assert [item["type"] for item in found] == ["text/csv", "image/png", "text/plain"]
        assert all(isinstance(item["modified"], float) for item in found)

    def test_what_is_hidden_and_what_is_linked_is_left_out(self, tmp_path: Path) -> None:
        write(tmp_path, "kept.txt")
        write(tmp_path, "uploads/.upload.abc.part")
        write(tmp_path, "__pycache__/mod.pyc")
        write(tmp_path, "notebooks/.ipynb_checkpoints/x.ipynb")
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        write(outside, "elsewhere.txt")
        (tmp_path / "link.txt").symlink_to(outside / "elsewhere.txt")
        (tmp_path / "folder").symlink_to(outside, target_is_directory=True)

        found, _ = list_files(tmp_path)

        assert [item["path"] for item in found] == ["kept.txt"]

    def test_a_listing_stops_at_its_limit_and_says_so(self, tmp_path: Path) -> None:
        for number in range(7):
            write(tmp_path, f"d{number % 2}/f{number}.txt")

        found, truncated = list_files(tmp_path, limit=5)

        assert truncated
        assert len(found) == 5
        assert [item["path"] for item in found] == sorted(item["path"] for item in found)

    def test_a_listing_that_has_reached_its_limit_does_not_go_on_through_the_rest_of_the_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("a", "b", "c", "d"):
            write(tmp_path, f"{name}/one.txt")
            write(tmp_path, f"{name}/two.txt")
        walked: list[str] = []
        real_walk = os.walk

        def watching_walk(*args: object, **kwargs: object):
            for step in real_walk(*args, **kwargs):  # type: ignore[call-overload]
                walked.append(step[0])
                yield step

        monkeypatch.setattr(files_module.os, "walk", watching_walk)

        found, truncated = list_files(tmp_path, limit=3)

        assert truncated
        assert len(found) == 3
        assert len(walked) <= 3

    def test_a_listing_exactly_at_its_limit_is_not_truncated(self, tmp_path: Path) -> None:
        for number in range(3):
            write(tmp_path, f"f{number}.txt")

        found, truncated = list_files(tmp_path, limit=3)

        assert len(found) == 3
        assert not truncated

    def test_only_the_parts_that_are_shown_are_listed(self, tmp_path: Path) -> None:
        write(tmp_path, "uploads/data.csv")
        write(tmp_path, "uploads/nested/more.csv")
        write(tmp_path, "chat.json")
        write(tmp_path, "messages.jsonl")
        write(tmp_path, "lab/meeting.json")
        write(tmp_path, "uploads_extra/data.csv")

        found, _ = list_files(tmp_path, shown=("uploads",))

        assert [item["path"] for item in found] == ["uploads/data.csv", "uploads/nested/more.csv"]

    def test_a_file_taken_away_while_the_listing_is_made_is_left_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(tmp_path, "real.txt")
        walked = [(str(tmp_path.resolve()), [], ["ghost.txt", "real.txt"])]
        monkeypatch.setattr(files_module.os, "walk", lambda *args, **kwargs: iter(walked))

        found, truncated = list_files(tmp_path)

        assert [item["path"] for item in found] == ["real.txt"]
        assert not truncated

    def test_an_empty_or_missing_root_lists_nothing(self, tmp_path: Path) -> None:
        assert list_files(tmp_path) == ([], False)
        assert list_files(tmp_path / "missing") == ([], False)


class TestTypes:
    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            ("plot.png", "image/png"),
            ("plot.svg", "image/svg+xml"),
            ("table.csv", "text/csv"),
            ("report.pdf", "application/pdf"),
            ("notes.md", "text/markdown"),
            ("archive.zip", "application/zip"),
            ("no_extension", "application/octet-stream"),
            ("weird.unknownext", "application/octet-stream"),
        ],
    )
    def test_a_file_is_sent_as_its_type(self, name: str, kind: str) -> None:
        assert media_type(Path(name)) == kind

    @pytest.mark.parametrize("name", ["page.html", "page.htm", "page.xhtml", "run.js", "run.mjs"])
    def test_what_a_browser_would_run_is_sent_as_text(self, name: str) -> None:
        assert media_type(Path(name)) == "text/plain"

    @pytest.mark.parametrize(
        "kind", ["text/html", "application/xhtml+xml", "text/javascript", "application/javascript"]
    )
    def test_every_type_a_browser_would_run_is_sent_as_text_whatever_this_system_calls_it(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(files_module.mimetypes, "guess_type", lambda name: (kind, None))

        assert media_type(Path("whatever")) == "text/plain"


class TestResponse:
    def test_a_picture_is_shown_in_the_page_and_nothing_in_it_may_run(self, tmp_path: Path) -> None:
        svg = write(tmp_path, "plot.svg", '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')

        response = file_response(svg)

        assert response.media_type == "image/svg+xml"
        assert response.headers["content-disposition"].startswith("inline")
        assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
        assert "sandbox" in CONTENT_SECURITY_POLICY
        assert "default-src 'none'" in CONTENT_SECURITY_POLICY
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_a_page_the_agents_wrote_is_shown_as_text_not_run(self, tmp_path: Path) -> None:
        page = write(tmp_path, "report.html", "<script>steal()</script>")

        response = file_response(page)

        assert response.media_type == "text/plain; charset=utf-8"
        assert response.headers["content-type"].startswith("text/plain")
        assert response.headers["content-disposition"].startswith("inline")

    def test_a_file_that_is_not_shown_in_a_page_is_saved(self, tmp_path: Path) -> None:
        archive = write(tmp_path, "results.zip")

        assert file_response(archive).headers["content-disposition"].startswith("attachment")

    def test_a_file_can_be_asked_for_to_be_saved(self, tmp_path: Path) -> None:
        plot = write(tmp_path, "plot.png")

        assert file_response(plot, download=True).headers["content-disposition"].startswith("attachment")

    def test_a_name_with_characters_a_header_cannot_hold_is_still_sent_under_its_name(self, tmp_path: Path) -> None:
        if os.name == "nt":
            pytest.skip("File names cannot hold these characters")
        odd = write(tmp_path, 'résumé "final".txt')

        disposition = file_response(odd).headers["content-disposition"]

        assert "filename*=utf-8''r%C3%A9sum%C3%A9%20%22final%22.txt" in disposition
