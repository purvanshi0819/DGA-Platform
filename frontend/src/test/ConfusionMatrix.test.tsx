import { render, screen } from "@testing-library/react";
import { ConfusionMatrix } from "../components/ConfusionMatrix";

it("renders counts and labels", () => {
  render(<ConfusionMatrix cm={{ labels: ["A", "B"], matrix: [[5, 1], [0, 7]] }} />);
  expect(screen.getByLabelText("Confusion matrix")).toBeInTheDocument();
  expect(screen.getByText("7")).toBeInTheDocument();
  expect(screen.getByText("n=6")).toBeInTheDocument();
});
