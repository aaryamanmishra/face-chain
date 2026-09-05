import json
import os
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()
w3 = Web3(Web3.HTTPProvider(os.getenv("RPC_URL")))

RESULT_JSON_PATH = "results/result.json"  # update this path if yours differs

def verify_from_result(result_path):
    with open(result_path, "r") as f:
        result = json.load(f)

    tx_hash = result["blockchain"]["sepolia_tx_hash"]
    expected_hash = result["evidence_hash"]

    if not tx_hash.startswith("0x"):
        tx_hash = "0x" + tx_hash

    tx = w3.eth.get_transaction(tx_hash)
    data = tx["input"]
    decoded = (
        data.decode("utf-8", errors="ignore")
        if isinstance(data, bytes)
        else bytes.fromhex(data[2:]).decode("utf-8", errors="ignore")
    )
    decoded = decoded.strip().lower().lstrip("0x")
    expected = expected_hash.strip().lower().lstrip("0x")

    etherscan_link = f"https://sepolia.etherscan.io/tx/{tx_hash}"

    print(f"TX Hash       : {tx_hash}")
    print(f"On-chain hash : {decoded}")
    print(f"Expected hash : {expected}")
    print(f"Etherscan     : {etherscan_link}")

    if decoded == expected:
        print("Result        : MATCH ✓")
    else:
        print("Result        : MISMATCH ✗")

    return decoded == expected

if __name__ == "__main__":
    verify_from_result(RESULT_JSON_PATH)