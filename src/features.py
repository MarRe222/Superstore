import sqlite3
import pandas as pd
import numpy as np

def load_cleaned_from_db(db_path: str) -> pd.DataFrame:
    """
    Load the cleaned Superstore dataset from SQLite.

    Reads the table `cleaned_superstore` created by the cleaning pipeline.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        DataFrame containing the cleaned data.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM cleaned_superstore", conn)
    conn.close()
    return df


def _add_profit_margin(df: pd.DataFrame) -> pd.DataFrame:

    """
    Add a profit margin feature.

    Computes profit margin as Profit / Sales for each row. Sales values of
    zero are replaced with NaN to avoid division errors.

    Args:
        df: Input DataFrame containing Sales and Profit columns.

    Returns:
        DataFrame with a new 'Profit Margin' column.
    """

    df = df.copy()
    df["Profit Margin"] = df["Profit"] / df["Sales"].replace(0, np.nan)
    return df

def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based date features.

    Extracts year, month, month name, and shipping duration from the
    Order Date and Ship Date columns.

    Features added:
        - Order Year
        - Order Month
        - Order Month Name
        - Ship Days

    Args:
        df: Input DataFrame with datetime columns 'Order Date' and 'Ship Date'.

    Returns:
        DataFrame with additional date-derived features.
    """

    df = df.copy()
    df["order_year"] = df["Order Date"].dt.year
    df["order_month"] = df["Order Date"].dt.month
    df["order_month_name"] = df["Order Date"].dt.strftime("%B")
    df["ship_days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df

def _add_yoy_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add year-over-year (YoY) sales growth.

    Computes total sales per year and calculates the percentage change
    relative to the previous year. The YoY growth value is merged back
    onto each row based on Order Year.

    Args:
        df: Input DataFrame containing 'order_year' and 'Sales'.

    Returns:
        DataFrame with a new 'YoY_Growth' column.
    """

    df = df.copy()
    year = (
        df.groupby("order_year")["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "sales_per_year"})
        .sort_values("order_year")
    )

    year["YoY_growth"] = year["sales_per_year"].pct_change().fillna(0)

    df = df.merge(
        year[["order_year", "YoY_growth"]],
        on="order_year",
        how="left"
    )
    return df
    
def _add_mom_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add month-over-month (MoM) sales growth.

    Aggregates sales by year and month, computes the percentage change
    from the previous month, and merges the result back onto the dataset.

    Args:
        df: Input DataFrame containing 'order_year', 'order_month', and 'Sales'.

    Returns:
        DataFrame with a new 'MoM_Growth' column.
    """

    df = df.copy()
    month = (
        df.groupby(["order_year","order_month"])["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "sales_per_month"})
        .sort_values(["order_year", "order_month"])
    )

    month["MoM_growth"] = month["sales_per_month"].pct_change()

    df = df.merge(
        month[["order_year","order_month", "MoM_growth"]],
        on=["order_year","order_month"],
        how="left"
    )
    return df
    

def _add_sales_per_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add total sales per order.

    Aggregates sales at the order level and merges the resulting
    Sales_per_Order value back onto each row.

    Args:
        df: Input DataFrame containing 'Order ID' and 'Sales'.

    Returns:
        DataFrame with a new 'Sales_per_Order' column.
    """

    df = df.copy()
    order_sales = (
        df.groupby("Order ID")["Sales"]
          .sum()
          .reset_index()
          .rename(columns={"Sales": "Sales_per_Order"})
    )

    df = df.merge(order_sales, on="Order ID", how="left")
    return df

def _add_total_sales_per_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add total lifetime sales per customer.

    Computes total sales for each customer across all orders and merges
    the result back onto the dataset.

    Args:
        df: Input DataFrame containing 'Customer ID' and 'Sales'.

    Returns:
        DataFrame with a new 'Customer_Sales' column.
    """

    df = df.copy()
    customer_sales = (
        df.groupby("Customer ID")["Sales"]
          .sum()
          .reset_index()
          .rename(columns={"Sales": "Customer_Sales"})
    )

    df = df.merge(customer_sales, on="Customer ID", how="left")
    return df


def _add_customer_lifetime_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add total lifetime profit per customer.

    Aggregates profit by customer and merges the resulting
    Customer_Profit value back onto each row.

    Args:
        df: Input DataFrame containing 'Customer ID' and 'Profit'.

    Returns:
        DataFrame with a new 'Customer_Profit' column.
    """

    df = df.copy()
    customer_profit = (
    df.groupby("Customer ID")["Profit"]
        .sum()
        .reset_index()
        .rename(columns={"Profit": "Customer_Profit"})
)

    df = df.merge(customer_profit, on="Customer ID", how="left")
    return df


def _add_order_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add total profit per order.

    Aggregates profit at the order level and merges the resulting
    Order_Profit value back onto each row.

    Args:
        df: Input DataFrame containing 'Order ID' and 'Profit'.

    Returns:
        DataFrame with a new 'Order_Profit' column.
    """

    df = df.copy()
    order_profit = (
    df.groupby("Order ID")["Profit"]
        .sum()
        .reset_index()
        .rename(columns={"Profit": "Order_Profit"})
)

    df = df.merge(order_profit, on="Order ID", how="left")
    return df

def _add_order_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add profit margin at the order level.

    Computes total profit and total sales per order, calculates the
    order-level margin, and merges it back onto the dataset.

    Args:
        df: Input DataFrame containing 'Order ID', 'Sales', and 'Profit'.

    Returns:
        DataFrame with a new 'Order_Margin' column.

    Notes:
        Requires both Sales and Profit to be non-zero for meaningful margins.
    """
    df = df.copy()
    order_margin = (
        df.groupby("Order ID")
          .apply(lambda x: x["Profit"].sum() / x["Sales"].sum())
          .rename("order_margin")
          .reset_index()
    )

    df = df.merge(order_margin, on="Order ID", how="left")

    return df

def _add_category_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add profit margin at the category level.

    Aggregates total sales and profit per category, computes category-level
    margin, and merges it back onto each row.

    Args:
        df: Input DataFrame containing 'Category', 'Sales', and 'Profit'.

    Returns:
        DataFrame with a new 'category_margin' column.
    """

    df = df.copy()
    category_margin = (
        df.groupby("Category")
          .agg(total_profit=("Profit", "sum"),
               total_sales=("Sales", "sum"))
          .reset_index()
    )

    category_margin["category_margin"] = (
        category_margin["total_profit"] / category_margin["total_sales"]
    )

    df = df.merge(
        category_margin[["Category", "category_margin"]],
        on="Category",
        how="left"
    )
    return df

def _add_category_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add each category's contribution to total sales.

    Computes total sales per category and expresses each category's share
    as a percentage of overall sales.

    Args:
        df: Input DataFrame containing 'Category' and 'Sales'.

    Returns:
        DataFrame with a new 'Category_Contribution' column.
    """

    df = df.copy()

    category_sales = (
        df.groupby("Category")["Sales"]
        .sum()
        .rename("category_total_sales")
        .reset_index()
    )
    total_sales = category_sales["category_total_sales"].sum()
    category_sales["category_contribution"] = (
        category_sales["category_total_sales"] / total_sales
    )

    df = df.merge(
        category_sales[["Category", "category_contribution"]],
        on="Category",
        how="left"
    )
    return df


def _add_region_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add each region's contribution to total sales.

    Computes total sales per region and merges the region-level contribution
    back onto the dataset.

    Args:
        df: Input DataFrame containing 'Region' and 'Sales'.

    Returns:
        DataFrame with a new 'region_contribution' column.
    """

    df = df.copy()
    region_sales = (
        df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )
    region_sales["region_contribution"] = (
        region_sales["Sales"] / region_sales["Sales"].sum()
    )
    df = df.merge(
        region_sales[["Region", "region_contribution"]],
        on="Region",
        how="left"
    )
    return df

def _add_high_discount_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary flag indicating high-discount rows.

    Flags rows where Discount exceeds 0.30.

    Args:
        df: Input DataFrame containing 'Discount'.

    Returns:
        DataFrame with a new 'high_discount' column (0 or 1).
    """

    df = df.copy()
    df["high_discount"] = (df["Discount"] > 0.3).astype(int)
    return df

def _add_discount_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a discount impact feature.

    Computes the monetary impact of discounting as Discount * Sales.

    Args:
        df: Input DataFrame containing 'Discount' and 'Sales'.

    Returns:
        DataFrame with a new 'Discount_Impact' column.
    """

    df = df.copy()
    df["discount_impact"] = df["Discount"] * df["Sales"]
    return df

def _add_profit_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    __________________________________________add profit ratio per unit.

    Computes profit per quantity sold for each row.

    Args:
        df: Input DataFrame containing 'Profit' and 'Quantity'.

    Returns:
        DataFrame with a new 'Profit_Ratio' column.
    """

    df = df.copy()
    df["profit_ratio"] = df["Profit"] / df["Quantity"]
    return df

def _add_order_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add total quantity per order.

    Aggregates quantity at the order level and merges the resulting
    Order_Size value back onto each row.

    Args:
        df: Input DataFrame containing 'Order ID' and 'Quantity'.

    Returns:
        DataFrame with a new 'Order_Size' column.
    """

    df = df.copy()
    order_size = (
        df.groupby("Order ID")["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "order_size"})
    )

    df = df.merge(order_size, on="Order ID", how="left")
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    return (
        df.copy()
        .pipe(_add_profit_margin)
        .pipe(_add_date_features)
        .pipe(_add_yoy_growth)
        .pipe(_add_mom_growth)
        .pipe(_add_sales_per_order)
        .pipe(_add_total_sales_per_customer)
        .pipe(_add_customer_lifetime_profit)
        .pipe(_add_order_profit)
        .pipe(_add_order_margin)
        .pipe(_add_category_margin)
        .pipe(_add_category_contribution)
        .pipe(_add_region_contribution)
        .pipe(_add_high_discount_flag)
        .pipe(_add_discount_impact)
        .pipe(_add_profit_ratio)
        .pipe(_add_order_size)
    )

def save_features_to_csv(df, output_path="data/processed/final_superstore.csv"):
    """
    Save the final feature-engineered dataset to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing engineered features.
    output_path : str
        File path where the CSV will be saved.
    """
    df.to_csv(output_path, index=False)

def run_features(db_path="database/superstore.db"):
    """
    Load cleaned data from SQLite, create engineered features,
    save the final dataset back to SQLite, and export to CSV.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    """

    # Load cleaned data
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM cleaned_superstore", conn)

    # Convert date columns (SQLite loads them as strings)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    # Create features
    df_features = create_features(df)

    # Save back to SQLite
    df_features.to_sql("final_superstore", conn, if_exists="replace", index=False)

    # Save to CSV
    save_features_to_csv(df_features)

    conn.close()

    print("Feature engineering complete. Saved to SQLite and CSV.")
