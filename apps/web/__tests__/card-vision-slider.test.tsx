/**
 * Tests for <CardVisionSlider>.
 *
 * Three behaviors that must hold:
 *   1. Both URLs present → slider renders, both <img> elements
 *      mount, the range input is interactive (covered by smoke
 *      events; we don't try to assert pixel-level opacity values
 *      since jsdom doesn't compute styles).
 *   2. Only frontUrl present → no slider, single static <img>.
 *   3. Both URLs null → component renders nothing (the parent
 *      cert page falls through to the placeholder rectangle).
 */
import { describe, it, expect, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import CardVisionSlider from "@/components/cert/CardVisionSlider";

describe("<CardVisionSlider>", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders both images and the slider when both URLs are present", () => {
    render(
      <CardVisionSlider
        frontUrl="https://example.com/front.png"
        flashUrl="https://example.com/flash.png"
        altPrefix="Pikachu V"
      />,
    );
    const imgs = screen.getAllByRole("img");
    expect(imgs).toHaveLength(2);
    expect(imgs[0]).toHaveAttribute("src", "https://example.com/front.png");
    expect(imgs[1]).toHaveAttribute("src", "https://example.com/flash.png");

    const slider = screen.getByTestId("card-vision-slider");
    expect(slider).toBeInTheDocument();
    expect(slider).toHaveAttribute("type", "range");
    expect(slider).toHaveAttribute(
      "aria-label",
      "Card Vision: crossfade between standard light and flash",
    );

    // Default position is 0% (regular front fully visible).
    expect(slider).toHaveValue("0");

    // Slider is keyboard/mouse interactive — moving it updates the
    // controlled value, which proves the React state wiring is live.
    fireEvent.change(slider, { target: { value: "75" } });
    expect(slider).toHaveValue("75");
  });

  it("renders only the front image and no slider when flashUrl is null", () => {
    render(
      <CardVisionSlider
        frontUrl="https://example.com/front.png"
        flashUrl={null}
      />,
    );
    const imgs = screen.getAllByRole("img");
    expect(imgs).toHaveLength(1);
    expect(imgs[0]).toHaveAttribute("src", "https://example.com/front.png");
    expect(screen.queryByTestId("card-vision-slider")).toBeNull();
  });

  it("renders nothing when both URLs are null (parent shows placeholder)", () => {
    const { container } = render(
      <CardVisionSlider frontUrl={null} flashUrl={null} />,
    );
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByRole("img")).toBeNull();
  });
});
