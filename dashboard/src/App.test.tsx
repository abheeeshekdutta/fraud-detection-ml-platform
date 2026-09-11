import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { fetchAlerts, fetchDecisions } from "./api";

vi.mock("./api", () => ({ fetchAlerts: vi.fn(), fetchDecisions: vi.fn() }));
afterEach(() => { cleanup(); vi.resetAllMocks(); });

describe("App", () => {
  it("shows an honest empty feed without fabricated transactions", async () => {
    vi.mocked(fetchDecisions).mockResolvedValue([]);
    vi.mocked(fetchAlerts).mockResolvedValue([]);
    render(<App />);
    expect(screen.getByText("Live transaction decisions")).toBeTruthy();
    await screen.findByText(/No scored transactions yet/);
    expect(screen.queryByText("2987000")).toBeNull();
    expect(screen.getByText("0 transactions")).toBeTruthy();
  });

  it("shows a service error instead of presenting failed requests as an empty feed", async () => {
    vi.mocked(fetchDecisions).mockRejectedValue(new Error("offline"));
    vi.mocked(fetchAlerts).mockResolvedValue([]);
    render(<App />);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Unable to refresh"));
    expect(screen.queryByText(/No scored transactions yet/)).toBeNull();
  });
});
