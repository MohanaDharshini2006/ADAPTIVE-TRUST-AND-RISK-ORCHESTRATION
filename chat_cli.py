"""Terminal test for the return conversation.   python chat_cli.py --customer CUST103
Attach a photo with:  /photo path\\to\\image.jpg     Show the structured result with:  /result     Quit: /quit
"""
import argparse
import json
import sys

sys.path.insert(0, "src")
from src.chat_agent import Session, handle

p = argparse.ArgumentParser()
p.add_argument("--customer", required=True, help="the signed-in customer's id, as stored in the orders")
s = Session(customer_id=p.parse_args().customer)
print("Agent:", handle(s, ""))
while True:
    line = input("You: ").strip()
    if line == "/quit":
        break
    if line == "/result":
        print(json.dumps(s.result, indent=2, default=str))
    elif line.startswith("/photo "):
        print("Agent:", handle(s, "", line[7:].strip().strip('"')))
    else:
        print("Agent:", handle(s, line))