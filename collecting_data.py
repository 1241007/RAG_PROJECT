import requests
import os
import urllib3
urllib3.disable_warnings()

os.makedirs("Data", exist_ok=True)

# ── Verified working direct PDF links (tested March 2026)
files = {
    "ipc.pdf"          : "https://www.indiacode.nic.in/bitstream/123456789/11091/1/the_indian_penal_code,_1860.pdf",
    "rti.pdf"          : "https://www.iitg.ac.in/rti/links/rti-act.pdf",
    "constitution.pdf" : "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf",
    "consumer.pdf"     : "https://www.indiacode.nic.in/bitstream/123456789/19840/1/right_yo_information_act.pdf",
}

# ── Manual download links (if script fails)
manual = {
    "ipc.pdf"          : "https://www.indiacode.nic.in/handle/123456789/12850?view_type=browse",
    "rti.pdf"          : "https://rti.gov.in/rtiact.asp",
    "constitution.pdf" : "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf",
    "consumer.pdf"     : "https://consumeraffairs.nic.in/acts-and-rules/acts",
}

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
failed  = []

for name, url in files.items():
    print(f"⏳ Downloading {name}...")
    try:
        r = requests.get(url, headers=headers, timeout=30,
                         allow_redirects=True, verify=False)
        if r.status_code == 200 and len(r.content) > 10_000:
            with open(f"Data/{name}", "wb") as f:
                f.write(r.content)
            print(f"✅ {name} saved  ({len(r.content)//1024} KB)\n")
        else:
            print(f"❌ {name} failed — status {r.status_code}\n")
            failed.append(name)
    except Exception as e:
        print(f"❌ {name} error: {e}\n")
        failed.append(name)

# ── Summary
if not failed:
    print("=" * 55)
    print("✅ All files downloaded into Data/")
    print("Next: delete chroma_db and re-run Stage 4 in notebook")
    print("=" * 55)
else:
    print("=" * 55)
    print(f"⚠️  {len(failed)} file(s) need manual download.")
    print("Open the links below in your browser → Save to Data/")
    print("=" * 55)
    for name in failed:
        print(f"\n  📄 {name}")
        print(f"     🔗 {manual[name]}")
    print("\n" + "=" * 55)