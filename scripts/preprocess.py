"""Clean and preprocess the four marketplace datasets.

Reads   raw/<platform>/<name>_raw.csv
Writes  preprocessed/<platform>/<name>_clean.csv

Run from the project root:  .venv/bin/python scripts/preprocess.py
"""

import ast
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
OUT = ROOT / "preprocessed"

# Shopee stores money as integers scaled by 100,000 (7500000 -> 75.00 TWD).
SHOPEE_PRICE_SCALE = 100_000

NO_BRAND = {"", "no brand", "nobrand", "no brands", "no", "none", "oem", "generic", "n/a", "na", "unbranded"}

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(s: pd.Series) -> pd.Series:
    """Strip HTML tags, collapse all whitespace (incl. newlines) and trim."""
    out = s.astype("string").str.replace(_TAG, " ", regex=True).str.replace(_WS, " ", regex=True).str.strip()
    return out.mask(out == "")


def clean_brand(s: pd.Series) -> pd.Series:
    out = clean_text(s)
    return out.mask(out.str.lower().isin(NO_BRAND))


def discount_pct(original: pd.Series, final: pd.Series) -> pd.Series:
    pct = (original - final) / original * 100
    return pct.where((original > 0) & (final <= original)).round(2)


def split_levels(df: pd.DataFrame, path: pd.Series, sep: str, levels: int = 3) -> None:
    parts = path.str.split(sep, regex=False)
    for i in range(levels):
        df[f"category_l{i + 1}"] = clean_text(parts.str[i])


def json_or_none(obj) -> str | None:
    return json.dumps(obj, ensure_ascii=False) if obj else None


# --------------------------------------------------------------------------- #
def flipkart() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "flipkart/flipkart_raw.csv")
    df = pd.DataFrame({"platform": "flipkart", "product_id": raw["pid"]}, index=raw.index)
    df["product_name"] = clean_text(raw["product_name"])
    df["brand"] = clean_brand(raw["brand"])

    # '["Clothing >> Women's Clothing >> ..."]' ; trees without '>>' are just a
    # truncated product name, not a category.
    tree = raw["product_category_tree"].str.replace(r'^\["|"\]$', "", regex=True)
    tree = tree.where(tree.str.contains(">>", regex=False))
    df["category_path"] = clean_text(tree.str.replace(">>", ">", regex=False))
    split_levels(df, tree, ">>")

    df["retail_price"] = raw["retail_price"]
    df["discounted_price"] = raw["discounted_price"]
    df["discount_pct"] = discount_pct(df["retail_price"], df["discounted_price"])
    df["currency"] = "INR"

    # product_rating and overall_rating are identical in the source; keep one.
    df["rating"] = pd.to_numeric(raw["product_rating"], errors="coerce")
    df["is_fk_advantage"] = raw["is_FK_Advantage_product"]
    df["description"] = clean_text(raw["description"])

    def specs(x):
        # Ruby-hash syntax: {"product_specification"=>[{"key"=>"Fabric", "value"=>"Cotton"}, ...]}
        if not isinstance(x, str):
            return None
        try:
            items = json.loads(x.replace("=>", ":").replace(":nil", ":null")).get("product_specification")
        except (json.JSONDecodeError, AttributeError):
            return None
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return None
        return json_or_none({i["key"].strip(): str(i.get("value", "")).strip()
                             for i in items if isinstance(i, dict) and i.get("key")})

    df["specifications"] = raw["product_specifications"].map(specs)

    def images(x):
        try:
            return json.loads(x)
        except (TypeError, json.JSONDecodeError):
            return []

    imgs = raw["image"].map(images)
    df["image_url"] = imgs.str[0]
    df["image_count"] = imgs.str.len()
    df["product_url"] = raw["product_url"]
    df["crawled_at"] = pd.to_datetime(raw["crawl_timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

    return df.drop_duplicates("product_id")


def amazon_2023() -> pd.DataFrame:
    """milistu/AMAZON-Products-2023 (30,000-row random sample, embeddings removed)."""
    raw = pd.read_csv(RAW / "amazon/amazon_2023_raw.csv")
    df = pd.DataFrame({"platform": "amazon", "product_id": raw["parent_asin"]}, index=raw.index)
    df["product_name"] = clean_text(raw["title"])
    # `store` sometimes carries a suffix: 'Scritti Politti   Format: Audio CD'.
    df["brand"] = clean_brand(raw["store"].str.replace(r"\s+Format:.*$", "", regex=True))

    def as_list(x):
        try:
            return [str(i).strip() for i in json.loads(x) if str(i).strip()]
        except (TypeError, json.JSONDecodeError):
            return []

    cats = raw["categories"].map(as_list)
    df["main_category"] = clean_text(raw["main_category"])
    df["category_path"] = cats.str.join(" > ").mask(cats.str.len() == 0)
    for i in range(3):
        df[f"category_l{i + 1}"] = cats.str[i]

    df["price"] = raw["price"].where(raw["price"] > 0)
    df["currency"] = "USD"
    df["rating"] = raw["average_rating"].where(raw["average_rating"].between(0, 5))
    df["rating_count"] = raw["rating_number"].astype("Int64")

    df["description"] = clean_text(raw["description"])
    feats = raw["features"].map(as_list)
    df["features"] = clean_text(feats.str.join(" | "))
    df["feature_count"] = feats.str.len()

    def details(x):
        # Stored as a Python dict repr: "{'Date First Available': 'April 29, 2023'}"
        try:
            d = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return None
        return json_or_none(d) if isinstance(d, dict) else None

    df["details"] = raw["details"].map(details)
    df["image_url"] = raw["image"]
    df["product_url"] = "https://www.amazon.com/dp/" + df["product_id"]
    df["first_available_date"] = pd.to_datetime(raw["date_first_available"], errors="coerce").dt.strftime("%Y-%m-%d")

    return df.drop_duplicates("product_id")


def lazada() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "lazada/lazada_raw.csv")
    # Dropped: `contents` (name + attributes + options concatenated for search)
    # and `product_product_id` (identical to `id`).
    df = pd.DataFrame({"platform": "lazada", "product_id": raw["id"]}, index=raw.index)
    df["product_name"] = clean_text(raw["product_product_name"])

    attrs = raw["product_attributes"].map(lambda x: json.loads(x) if isinstance(x, str) else {})
    df["brand"] = clean_brand(attrs.map(lambda a: (a.get("brand") or [None])[0]))

    df["category_path"] = clean_text(raw["product_category"].str.replace(" - ", " > ", regex=False))
    split_levels(df, raw["product_category"], " - ")

    df["price"] = raw["product_price"]
    df["seller_id"] = raw["product_seller_id"]
    df["seller_name"] = clean_text(raw["product_seller_name"])
    df["description"] = clean_text(raw["product_description"])
    df["attributes"] = attrs.map(lambda a: json_or_none({k: v for k, v in a.items() if k != "brand"}))

    opts = raw["product_options"].map(lambda x: [o for o in json.loads(x) if o] if isinstance(x, str) else [])
    df["options"] = opts.map(json_or_none)
    df["option_count"] = opts.str.len()

    return df.drop_duplicates("product_id")


def shopee() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "shopee/shopee_raw.csv", low_memory=False)
    # Dropped: price, priceBeforeDiscount, priceMin/MaxBeforeDiscount, models,
    # productUrl (all redacted to '[PREMIUM]'), isIndividualSeller, isMart
    # (100% empty) and _primaryKey (float-rounded itemId/shopId).
    df = pd.DataFrame({"platform": "shopee", "product_id": raw["itemId"]}, index=raw.index)
    df["product_name"] = clean_text(raw["title"])
    df["brand"] = clean_brand(raw["brand"])

    def cat_names(x):
        try:
            return [json.loads(c)["display_name"] for c in eval(x, {"__builtins__": {}})]  # noqa: S307
        except Exception:
            return re.findall(r'"display_name":"([^"]*)"', str(x))

    # Some display names carry an embedded line break ('電源插座/\n延長線').
    names = raw["categories"].map(lambda x: [_WS.sub("", n) if "\n" in n else n.strip() for n in cat_names(x)])
    df["category_id"] = raw["catId"]
    df["category_path"] = names.str.join(" > ")
    for i in range(3):
        df[f"category_l{i + 1}"] = names.str[i]

    df["price_min"] = raw["priceMin"] / SHOPEE_PRICE_SCALE
    df["price_max"] = raw["priceMax"] / SHOPEE_PRICE_SCALE
    df["discount_pct"] = raw["discount"]
    df["currency"] = raw["currency"]

    # A 0.0 average with no ratings means "unrated", not "zero stars".
    df["rating"] = raw["ratingAvg"].where(raw["ratingCount"] > 0).round(2)
    df["rating_count"] = raw["ratingCount"]
    df["comment_count"] = raw["commentCount"]
    df["liked_count"] = raw["likedCount"]

    df["status"] = raw["status"]
    df["condition_code"] = raw["condition"]
    df["variant_count"] = raw["modelsCount"]
    df["image_count"] = raw["images"].str.count("'") // 2
    df["is_adult"] = raw["isAdult"]
    df["is_pre_order"] = raw["isPreOrder"]

    df["estimated_delivery_days"] = raw["estimatedDays"]
    df["is_free_shipping"] = raw["isFreeShipping"]
    df["is_service_by_shopee"] = raw["isServiceByShopee"]
    # -1 is the source's "unknown" sentinel.
    df["shipping_fee_min"] = raw["shippingFeeMin"].where(raw["shippingFeeMin"] >= 0) / SHOPEE_PRICE_SCALE

    df["seller_id"] = raw["shopId"]
    df["seller_name"] = clean_text(raw["shopName"])
    df["seller_location"] = clean_text(raw["shopDetailedLocation"].fillna(raw["shopLocation"]))
    df["seller_rating"] = raw["shopRating"].round(3)
    df["seller_rating_good"] = raw["shopRatingGood"]
    df["seller_rating_normal"] = raw["shopRatingNormal"]
    df["seller_rating_bad"] = raw["shopRatingBad"]
    df["seller_response_rate"] = raw["shopResponseRate"]
    df["seller_response_time_sec"] = raw["shopResponseTime"]
    df["seller_follower_count"] = raw["shopFollowerCount"]
    df["seller_item_count"] = raw["shopItemCount"]
    df["is_official_shop"] = raw["isOfficialShop"]
    df["is_shopee_verified"] = raw["isShopeeVerified"]

    df["description"] = clean_text(raw["description"])
    for new, old in [("created_at", "createdAt"), ("seller_created_at", "shopCreatedAt"),
                     ("first_seen_at", "_firstSeenAt"), ("last_seen_at", "_lastSeenAt")]:
        df[new] = pd.to_datetime(raw[old], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    return df.drop_duplicates("product_id")


# --------------------------------------------------------------------------- #
def main() -> None:
    jobs = [("flipkart", "flipkart", flipkart), ("amazon", "amazon_2023", amazon_2023),
            ("lazada", "lazada", lazada), ("shopee", "shopee", shopee)]
    for folder, name, fn in jobs:
        df = fn()
        before = len(df)
        df = df.dropna(subset=["product_name"]).reset_index(drop=True)
        dest = OUT / folder / f"{name}_clean.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        print(f"{name:11s} {len(df):6d} rows x {df.shape[1]:2d} cols  "
              f"(dropped {before - len(df)} unnamed)  -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
