# ==========================================================
# STOCK SCREENER PROJECT — Quant Guild | Command-Line Version
# ==========================================================
# SPECIFICATIONS:
# - No external imports or modules are used.
# - No type hints or annotations.
# - Uses only built-in data structures (list, dict, string, float).
# - Provides interactive CLI to manage filters and screen stocks.
# - Commands: ADD, REMOVE, SHOW FILTERS, SCREEN, CLEAR, EXIT.
# ==========================================================

# ----------------------------------------------------------
# Pre-Populated Dataset (List of Dictionaries)
# Each dictionary represents a stock with six required metrics.
# ----------------------------------------------------------
market_data = [
    {'Ticker': 'AAPL', 'Sector': 'Technology', 'Price': 175.50, 'MarketCap_B': 2800.0, 'PE_Ratio': 30.1, 'DividendYield': 0.55},
    {'Ticker': 'MSFT', 'Sector': 'Technology', 'Price': 420.20, 'MarketCap_B': 3100.0, 'PE_Ratio': 35.5, 'DividendYield': 0.70},
    {'Ticker': 'JPM',  'Sector': 'Finance', 'Price': 190.80, 'MarketCap_B': 570.0, 'PE_Ratio': 12.5, 'DividendYield': 2.30},
    {'Ticker': 'VZ',   'Sector': 'Communication', 'Price': 40.10, 'MarketCap_B': 170.0, 'PE_Ratio': 8.8, 'DividendYield': 6.50},
    {'Ticker': 'TSLA', 'Sector': 'Automotive', 'Price': 205.00, 'MarketCap_B': 650.0, 'PE_Ratio': 62.0, 'DividendYield': 0.00},
    {'Ticker': 'SBUX', 'Sector': 'Consumer Cyclical', 'Price': 95.00, 'MarketCap_B': 105.0, 'PE_Ratio': 25.4, 'DividendYield': 2.10},
    {'Ticker': 'HD',   'Sector': 'Consumer Cyclical', 'Price': 350.00, 'MarketCap_B': 360.0, 'PE_Ratio': 21.9, 'DividendYield': 2.45},
    {'Ticker': 'PFE',  'Sector': 'Healthcare', 'Price': 28.50, 'MarketCap_B': 160.0, 'PE_Ratio': 15.0, 'DividendYield': 5.50},
    {'Ticker': 'XOM',  'Sector': 'Energy', 'Price': 115.00, 'MarketCap_B': 470.0, 'PE_Ratio': 10.2, 'DividendYield': 3.80},
    {'Ticker': 'KO',   'Sector': 'Consumer Defensive', 'Price': 63.00, 'MarketCap_B': 270.0, 'PE_Ratio': 23.5, 'DividendYield': 3.10},
    {'Ticker': 'GS',   'Sector': 'Finance', 'Price': 400.00, 'MarketCap_B': 130.0, 'PE_Ratio': 11.1, 'DividendYield': 2.70},
    {'Ticker': 'NFLX', 'Sector': 'Communication', 'Price': 625.00, 'MarketCap_B': 265.0, 'PE_Ratio': 45.0, 'DividendYield': 0.00},
    {'Ticker': 'AMZN', 'Sector': 'Consumer Cyclical', 'Price': 180.00, 'MarketCap_B': 1850.0, 'PE_Ratio': 40.0, 'DividendYield': 0.00},
    {'Ticker': 'ORCL', 'Sector': 'Technology', 'Price': 130.00, 'MarketCap_B': 365.0, 'PE_Ratio': 32.2, 'DividendYield': 1.40},
    {'Ticker': 'WFC',  'Sector': 'Finance', 'Price': 60.00, 'MarketCap_B': 200.0, 'PE_Ratio': 10.5, 'DividendYield': 2.90}
]

# ----------------------------------------------------------
# Lists of field types (used for validation)
# ----------------------------------------------------------
numeric_fields = ["Price", "MarketCap_B", "PE_Ratio", "DividendYield"]
categorical_fields = ["Ticker", "Sector"]

# Active filters will be stored here as a list of tuples:
# Example: ('Price', '<', 100.0, 'numeric')
filters = []

# ----------------------------------------------------------
# FUNCTION: parse_filter
# Purpose: Converts a raw input string like "Price < 100"
#          into a structured filter tuple.
# ----------------------------------------------------------
def parse_filter(expr):
    parts = expr.strip().split()

    # Must have at least: FIELD OPERATOR VALUE
    if len(parts) < 3:
        print("Invalid filter format. Use: Field Operator Value")
        return None

    field = parts[0]
    op = parts[1]
    value = " ".join(parts[2:])  # handles multi-word sectors like "Consumer Cyclical"

    # Validate field existence
    if field not in numeric_fields + categorical_fields:
        print("Unknown field:", field)
        return None

    # Handle numeric filters
    if field in numeric_fields:
        try:
            value = float(value)
        except:
            print("Numeric value expected for", field)
            return None
        if op not in ["<", "<=", ">", ">=", "=", "=="]:
            print("Invalid operator for numeric field.")
            return None
        if op == "==": op = "="
        return (field, op, value, "numeric")

    # Handle categorical filters
    else:
        if op not in ["=", "=="]:
            print("Only '=' allowed for categorical fields.")
            return None
        if op == "==": op = "="
        return (field, op, value, "categorical")

# ----------------------------------------------------------
# FUNCTION: apply_filters
# Purpose: Applies all active filters (AND logic) to the dataset.
# ----------------------------------------------------------
def apply_filters(data, filters):
    result = []
    for stock in data:
        keep = True  # assume stock passes all filters
        for f in filters:
            field, op, value, kind = f

            # For numeric comparisons
            if kind == "numeric":
                sval = stock[field]
                if op == "<" and not (sval < value): keep = False
                elif op == "<=" and not (sval <= value): keep = False
                elif op == ">" and not (sval > value): keep = False
                elif op == ">=" and not (sval >= value): keep = False
                elif op == "=" and not (sval == value): keep = False

            # For categorical comparisons (exact match)
            else:
                if str(stock[field]) != str(value):
                    keep = False

            # If any filter fails, stop checking further
            if not keep:
                break

        # Keep stock only if all filters matched
        if keep:
            result.append(stock)
    return result

# ----------------------------------------------------------
# FUNCTION: show_filters
# Purpose: Displays all active filters with numbering.
# ----------------------------------------------------------
def show_filters():
    if not filters:
        print("No active filters.")
    else:
        print("Active Filters (" + str(len(filters)) + "):")
        i = 1
        for f in filters:
            print(str(i) + ".", f[0], f[1], f[2])
            i += 1

# ----------------------------------------------------------
# FUNCTION: screen
# Purpose: Displays filtered results in a tabular format.
# ----------------------------------------------------------
def screen():
    # If no filters are active, show all data
    if not filters:
        print("No filters. Showing all stocks.")
        data = market_data
    else:
        print("Screening", len(market_data), "stocks...")
        data = apply_filters(market_data, filters)

    # If nothing matched
    if not data:
        print("No stocks matched your criteria.")
        return

    # Columns to display in table
    cols = ["Ticker", "Sector", "Price", "MarketCap_B", "PE_Ratio", "DividendYield"]

    # Build header and separator lines
    header = "| " + " | ".join(cols) + " |"
    sep = "|-" + "-|-".join(["-" * len(c) for c in cols]) + "-|"

    print(header)
    print(sep)

    # Print each stock row
    for d in data:
        row = []
        for c in cols:
            val = d[c]
            # Format float values to 2 decimals for neatness
            if isinstance(val, float):
                val = "%.2f" % val
            row.append(str(val))
        print("| " + " | ".join(row) + " |")

    print("(" + str(len(data)) + " stocks found)")

# ----------------------------------------------------------
# FUNCTION: main
# Purpose: Handles user input loop and program flow.
# ----------------------------------------------------------
def main():
    print("Welcome to Quant Guild Stock Screener. Enter 'help' for options.")
    while True:
        cmd = input("> ").strip()  # user command input

        # Skip empty inputs
        if cmd == "":
            continue

        # Convert to lowercase for easier comparison
        low = cmd.lower()

        # Exit program
        if low == "exit":
            print("Thank you for using the screener. Goodbye!")
            break

        # Help menu
        elif low == "help":
            print("Commands:")
            print("  ADD <Field> <Op> <Value>      -> Add a new filter")
            print("  REMOVE <index> or <filter>    -> Remove a filter")
            print("  SHOW FILTERS                  -> Display all active filters")
            print("  SCREEN                        -> Apply filters and show results")
            print("  CLEAR                         -> Remove all filters")
            print("  EXIT                          -> Quit the program")

        # Add a new filter
        elif low.startswith("add "):
            expr = cmd[4:].strip()
            f = parse_filter(expr)
            if f:
                filters.append(f)
                print("Filter added:", f[0], f[1], f[2])

        # Remove a filter
        elif low.startswith("remove "):
            arg = cmd[7:].strip()
            if not filters:
                print("No filters to remove.")
                continue

            # Remove by index
            if arg.isdigit():
                i = int(arg)
                if 1 <= i <= len(filters):
                    removed = filters.pop(i - 1)
                    print("Filter removed:", removed[0], removed[1], removed[2])
                else:
                    print("Invalid index.")
            # Remove by retyping the filter
            else:
                f = parse_filter(arg)
                if f and f in filters:
                    filters.remove(f)
                    print("Filter removed:", f[0], f[1], f[2])
                else:
                    print("No matching filter found.")

        # Show all filters
        elif low == "show filters":
            show_filters()

        # Clear all filters
        elif low == "clear":
            filters.clear()
            print("All filters cleared.")

        # Run the screen command
        elif low == "screen":
            screen()

        # Invalid command
        else:
            print("Unknown command. Type 'help' for options.")

# ----------------------------------------------------------
# PROGRAM ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
