import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "./App";

describe("App", () => {
  it("renders the operations dashboard", () => {
    render(<App />);

    expect(screen.getByText("Live transaction decisions")).toBeTruthy();
    expect(screen.getByText("Fraud operations")).toBeTruthy();
    expect(screen.getByText("Decision feed")).toBeTruthy();
  });
});
