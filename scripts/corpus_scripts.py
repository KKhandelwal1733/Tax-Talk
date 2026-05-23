"""Corpus source URLs and download notes.

All sources below are official Indian government publications (public documents).
Run this script for guidance — actual download is manual to ensure you grab the
latest versions and respect rate limits.
"""

CORPUS_SOURCES = {
    "cgst_act": {
        "url": "https://cbic-gst.gov.in/CGST-bill-e.html",
        "format": "PDF",
        "notes": "Central Goods and Services Tax Act, 2017 + amendments through latest Finance Act",
    },
    "igst_act": {
        "url": "https://cbic-gst.gov.in/IGST-bill-e.html",
        "format": "PDF",
        "notes": "Integrated GST Act, 2017",
    },
    "ugst_act": {
        "url": "https://cbic-gst.gov.in/UTGST-bill-e.html",
        "format": "PDF",
        "notes": "Union Territory GST Act, 2017",
    },
    "income_tax_act_1961": {
        "url": "https://incometaxindia.gov.in/Pages/acts/income-tax-act.aspx",
        "format": "PDF",
        "notes": "Income Tax Act 1961, as amended. Use the latest 'consolidated' version.",
    },
    "gst_notifications": {
        "url": "https://www.cbic.gov.in/entities/cbic-content-mst/MTQ4MTU=",
        "format": "PDF + HTML",
        "notes": "CBIC notifications — start with last 2 years, expand if needed.",
    },
    "income_tax_notifications": {
        "url": "https://incometaxindia.gov.in/pages/communications/notifications.aspx",
        "format": "PDF",
        "notes": "Income Tax Department notifications.",
    },
    "gst_circulars": {
        "url": "https://www.cbic.gov.in/entities/cbic-content-mst/MTQ4MjY=",
        "format": "PDF",
        "notes": "CBIC circulars — interpretive guidance, valuable for nuanced questions.",
    },
    "egazette": {
        "url": "https://egazette.gov.in/",
        "format": "PDF",
        "notes": "Official gazette — Finance Acts, ordinances.",
    },
}


def main() -> None:
    print("📚 Corpus sources for Tax Talk App (GST + IncomTe Tax)")
    print("=" * 70)
    for name, meta in CORPUS_SOURCES.items():
        print(f"\n## {name}")
        print(f"  URL:    {meta['url']}")
        print(f"  Format: {meta['format']}")
        print(f"  Notes:  {meta['notes']}")

    print("\n" + "=" * 70)
    print("Download into data/raw/<source_name>/. Do NOT commit the data files.")
    print("Aim for ~50–200 MB of corpus in week 3. Expand iteratively.")


if __name__ == "__main__":
    main()
