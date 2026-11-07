"""Hand-written synthetic receipt fixtures, not records from the CORD dataset."""

from __future__ import annotations

from dataclasses import dataclass

from qlora_extraction.schema import ReceiptItem, ReceiptSchema


@dataclass(frozen=True)
class ReceiptDocument:
    """Document record with raw OCR transcript and target structured schema."""

    doc_id: str
    ocr_text: str
    ground_truth: ReceiptSchema
    split: str = "test"  # "train", "val", "test"


def load_cord_dataset() -> list[ReceiptDocument]:
    """Load five synthetic demo receipts across train, val, and test splits."""
    return [
        ReceiptDocument(
            doc_id="synthetic_001",
            ocr_text=(
                "STARBUCKS COFFEE #1042\n"
                "123 PIKE STREET, SEATTLE WA\n"
                "DATE: 2026-03-15 08:32 AM\n"
                "1 CAFE LATTE GRANDE   $4.95\n"
                "1 BLUEBERRY MUFFIN     $3.25\n"
                "SUBTOTAL               $8.20\n"
                "TAX 8.5%               $0.70\n"
                "TOTAL                  $8.90\n"
                "VISA CARD ****1234"
            ),
            ground_truth=ReceiptSchema(
                merchant_name="Starbucks Coffee",
                transaction_date="2026-03-15",
                currency="USD",
                subtotal=8.20,
                tax=0.70,
                total=8.90,
                line_items=[
                    ReceiptItem(description="Cafe Latte Grande", quantity=1, price=4.95),
                    ReceiptItem(description="Blueberry Muffin", quantity=1, price=3.25),
                ],
            ),
            split="train",
        ),
        ReceiptDocument(
            doc_id="synthetic_002",
            ocr_text=(
                "TARGET STORE T-0891\n"
                "MINNEAPOLIS, MN\n"
                "DT: 2026-03-18\n"
                "NOTEBOOK SPIRAL 3PK    $6.50\n"
                "GEL PENS BLACK 5PK     $4.99\n"
                "TAX                     $0.80\n"
                "TOTAL AMOUNT           $12.29\n"
                "AUTH: 994821"
            ),
            ground_truth=ReceiptSchema(
                merchant_name="Target",
                transaction_date="2026-03-18",
                currency="USD",
                subtotal=11.49,
                tax=0.80,
                total=12.29,
                line_items=[
                    ReceiptItem(description="Notebook Spiral 3PK", quantity=1, price=6.50),
                    ReceiptItem(description="Gel Pens Black 5PK", quantity=1, price=4.99),
                ],
            ),
            split="train",
        ),
        ReceiptDocument(
            doc_id="synthetic_003",
            ocr_text=(
                "WHOLE FOODS MARKET\n"
                "AUSTIN TX\n"
                "TRANSACTION DATE: 2026-04-02\n"
                "ORGANIC APPLES 2LB     $5.98\n"
                "ALMOND MILK 64OZ       $3.49\n"
                "SOURDOUGH BREAD        $4.50\n"
                "SUBTOTAL               $13.97\n"
                "TAX                     $0.55\n"
                "TOTAL DUE              $14.52\n"
                "THANK YOU FOR SHOPPING"
            ),
            ground_truth=ReceiptSchema(
                merchant_name="Whole Foods Market",
                transaction_date="2026-04-02",
                currency="USD",
                subtotal=13.97,
                tax=0.55,
                total=14.52,
                line_items=[
                    ReceiptItem(description="Organic Apples 2LB", quantity=1, price=5.98),
                    ReceiptItem(description="Almond Milk 64OZ", quantity=1, price=3.49),
                    ReceiptItem(description="Sourdough Bread", quantity=1, price=4.50),
                ],
            ),
            split="val",
        ),
        ReceiptDocument(
            doc_id="synthetic_004",
            ocr_text=(
                "CHIPOTLE MEXICAN GRILL #231\n"
                "NEW YORK, NY\n"
                "DATE: 2026-04-10 12:15 PM\n"
                "BURRITO BOWL CHICKEN   $10.75\n"
                "CHIPS & GUACAMOLE      $4.25\n"
                "FOUNTAIN SODA          $2.50\n"
                "TAX 8.875%             $1.55\n"
                "TOTAL                  $19.05\n"
                "MC ****5678"
            ),
            ground_truth=ReceiptSchema(
                merchant_name="Chipotle Mexican Grill",
                transaction_date="2026-04-10",
                currency="USD",
                subtotal=17.50,
                tax=1.55,
                total=19.05,
                line_items=[
                    ReceiptItem(description="Burrito Bowl Chicken", quantity=1, price=10.75),
                    ReceiptItem(description="Chips & Guacamole", quantity=1, price=4.25),
                    ReceiptItem(description="Fountain Soda", quantity=1, price=2.50),
                ],
            ),
            split="test",
        ),
        ReceiptDocument(
            doc_id="synthetic_005",
            ocr_text=(
                "SHELL GAS STATION #4910\n"
                "HOUSTON, TX\n"
                "DATE 2026-04-12\n"
                "REGULAR UNLEADED 10GAL $32.50\n"
                "CAR WASH DELUXE        $10.00\n"
                "TOTAL SALE             $42.50\n"
                "PUMP #04 APPROVED"
            ),
            ground_truth=ReceiptSchema(
                merchant_name="Shell Gas Station",
                transaction_date="2026-04-12",
                currency="USD",
                subtotal=42.50,
                tax=0.00,
                total=42.50,
                line_items=[
                    ReceiptItem(description="Regular Unleaded 10GAL", quantity=1, price=32.50),
                    ReceiptItem(description="Car Wash Deluxe", quantity=1, price=10.00),
                ],
            ),
            split="test",
        ),
    ]
