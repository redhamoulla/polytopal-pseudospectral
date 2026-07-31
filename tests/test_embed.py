from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EMBED = ROOT / "docs" / "embed.html"


class _EmbedParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.iframes: list[dict[str, str]] = []
        self.links: list[dict[str, str]] = []
        self.buttons: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "iframe":
            self.iframes.append(values)
        elif tag == "a":
            self.links.append(values)
        elif tag == "button":
            self.buttons.append(values)


def test_embed_has_single_same_origin_frame_and_click_through_link() -> None:
    source = EMBED.read_text(encoding="utf-8")
    parser = _EmbedParser()
    parser.feed(source)

    assert len(parser.iframes) == 1
    assert parser.iframes[0]["src"] == "./index.html"
    assert parser.iframes[0]["aria-hidden"] == "true"
    assert parser.iframes[0]["tabindex"] == "-1"

    assert len(parser.links) == 1
    assert parser.links[0]["href"] == "./"
    assert parser.links[0]["target"] == "_blank"
    assert set(parser.links[0]["rel"].split()) == {"noopener", "noreferrer"}
    assert "new tab" in parser.links[0]["aria-label"].lower()

    assert len(parser.buttons) == 1
    assert parser.buttons[0]["type"] == "button"
    assert parser.buttons[0]["aria-label"] == "Pause animation"
    assert "hidden" in parser.buttons[0]


def test_embed_autoplays_safely_and_honours_reduced_motion() -> None:
    source = EMBED.read_text(encoding="utf-8")

    assert 'window.matchMedia("(prefers-reduced-motion: reduce)")' in source
    assert "document.hidden" in source
    assert "playButton.disabled = false" in source
    assert "playButton.click()" in source
    assert "window.setTimeout(runWave, 8600)" in source
    assert 'frame.addEventListener("load", prepareFrame, { once: true })' in source
    assert "cancelActiveFrames()" in source
    assert 'animationToggle.addEventListener("click"' in source
    assert '"IntersectionObserver" in window' in source
    assert "pointer-events: none" in source
    assert "width: min(100vw, 200vh)" in source
    assert "height: min(100vh, 50vw)" in source
    assert ".page-shell" in source
    assert "height: 100% !important" in source
    assert ".pw-energy" in source
    assert ".pw-learning" in source
