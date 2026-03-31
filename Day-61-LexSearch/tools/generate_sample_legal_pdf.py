from __future__ import annotations

from pathlib import Path


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_simple_pdf(lines: list[str]) -> bytes:
    content_lines = [
        "BT",
        "/F1 11 Tf",
        "1 0 0 1 48 770 Tm",
        "14 TL",
    ]

    for i, line in enumerate(lines):
        if i == 0:
            content_lines.append(f"({escape_pdf_text(line)}) Tj")
        else:
            content_lines.append("T*")
            content_lines.append(f"({escape_pdf_text(line)}) Tj")

    content_lines.append("ET")
    content_stream = ("\n".join(content_lines) + "\n").encode("utf-8")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>\nstream\n"
        + content_stream
        + b"endstream",
    ]

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n")

    offsets = [0]
    for obj_num, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_num} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_start}\n"
            "%%EOF\n"
        ).encode("ascii")
    )

    return bytes(pdf)


def build_legal_lines() -> list[str]:
    return [
        "IN THE HIGH COURT OF NEW DELHI AT NEW DELHI",
        "CIVIL ORIGINAL JURISDICTION",
        "Case No. 245 of 2024",
        "Apex Infra Developers Pvt. Ltd. v. Northline Commercial Leasing LLP",
        "JUDGMENT (Extract for Research Use)",
        "Facts: The parties executed a commercial lease agreement on 03 January 2022.",
        "Clause 14 required rent payment by the 7th day of each month.",
        "Clause 19 provided liquidated damages of INR 4,00,000 for material breach.",
        "The tenant stopped payment from August 2023 and vacated without notice.",
        "Issue 1: Whether non payment amounted to a fundamental breach of contract.",
        "Issue 2: Whether liquidated damages clause was enforceable under Indian law.",
        "Issue 3: Whether specific performance could be granted in the circumstances.",
        "Findings: The Court held there was a breach of contract and wrongful termination.",
        "The landlord proved actual loss through vacancy records and brokerage invoices.",
        "The liquidated damages clause was treated as a genuine pre estimate of loss.",
        "The Court cited Section 74 of the Indian Contract Act, 1872.",
        "The Court also distinguished penalties from enforceable liquidated damages.",
        "Precedent discussed: Kailash Nath Associates v. DDA (2015) 4 SCC 136.",
        "Precedent discussed: ONGC v. Saw Pipes Ltd. (2003) 5 SCC 705.",
        "Relief: Damages of INR 4,00,000 and unpaid rent with 9 percent annual interest.",
        "Prayer for specific performance was denied because tenancy had already ended.",
        "Observation: Arbitration clause did not bar urgent civil relief for possession.",
        "Date of Decision: 18 November 2024.",
        "Coram: Justice R. Menon.",
        "Keywords: breach of contract, liquidated damages, lease default, specific performance.",
    ]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "test-data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "legal_sample_case.pdf"
    pdf_bytes = build_simple_pdf(build_legal_lines())
    output_path.write_bytes(pdf_bytes)

    print(f"Created sample legal PDF at: {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
