import { describe, expect, it } from "vitest";
import { checkGas, toCsv, pct } from "../utils/format";

describe("checkGas", () => {
  it("rejects missing required values instead of defaulting to zero", () => {
    expect(checkGas("", true).error).toMatch(/Required/);
    expect(checkGas("  ", true).value).toBeUndefined();
  });
  it("allows an empty optional field", () => expect(checkGas("", false)).toEqual({}));
  it("rejects non-numeric and negative values", () => {
    expect(checkGas("abc", true).error).toBeDefined();
    expect(checkGas("-1", true).error).toMatch(/negative/);
  });
  it("accepts decimals and exponents", () => {
    expect(checkGas("12.5", true).value).toBe(12.5);
    expect(checkGas("1e3", true).value).toBe(1000);
  });
});
describe("utils", () => {
  it("escapes csv cells", () => expect(toCsv([{ a: 'x,"y"' }])).toBe('a\n"x,""y"""'));
  it("formats percentages", () => expect(pct(0.9123)).toBe("91.2%"));
});
