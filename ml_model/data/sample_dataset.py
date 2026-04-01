"""
sample_dataset.py
-----------------
Generates a labelled sample dataset for training and evaluation.

Labels
------
  scam     : 1
  legitimate: 0

Sentiment categories (scammer sentiment)
-----------------------------------------
  urgent    : high-pressure, fear-driven language
  greedy    : lure of money / prize
  friendly  : false trust-building
  neutral   : no strong emotional tone
"""

import pandas as pd
import random

SCAM_MESSAGES = [
    # Urgent / fear-driven
    ("URGENT: Your account has been compromised. Click here immediately to verify your identity or your account will be suspended!", 1, "urgent"),
    ("WARNING! We detected unauthorized login from Russia. Confirm your details NOW to avoid losing access.", 1, "urgent"),
    ("Your bank account will be CLOSED in 24 hours unless you update your information. Act immediately!", 1, "urgent"),
    ("FINAL NOTICE: You owe outstanding taxes. Failure to pay today will result in arrest.", 1, "urgent"),
    ("Your Social Security Number has been suspended. Call us immediately to avoid legal consequences.", 1, "urgent"),
    ("ALERT: Your computer is infected with a virus. Call Microsoft Support NOW at 1-800-555-0199.", 1, "urgent"),
    ("Immediate action required: Your PayPal account is limited. Verify now or lose your funds.", 1, "urgent"),
    ("You have 2 hours to respond or your account will be permanently deleted. Click link to confirm.", 1, "urgent"),

    # Greedy / prize lure
    ("Congratulations! You have won $1,000,000 in our international lottery. Send your bank details to claim.", 1, "greedy"),
    ("You are selected as today's lucky winner! Claim your free iPhone 15 Pro by clicking this link.", 1, "greedy"),
    ("A wealthy Nigerian prince needs your help transferring $45 million. You will receive 30% for helping.", 1, "greedy"),
    ("Your email was selected in our random prize draw. You've won $500 Amazon gift card. Collect now!", 1, "greedy"),
    ("Invest just $100 today and make $10,000 in 7 days. Guaranteed returns. Join our crypto group.", 1, "greedy"),
    ("We found unclaimed inheritance money in your name worth $2.4 million. Contact us to retrieve it.", 1, "greedy"),
    ("You have been pre-approved for a $50,000 loan with no credit check. Apply in minutes, cash same day.", 1, "greedy"),
    ("Work from home and earn $5000 per week with no experience needed. Limited spots available!", 1, "greedy"),

    # Friendly / trust-building
    ("Hi, this is Sarah from customer support. I noticed your account and wanted to personally help you.", 1, "friendly"),
    ("Hey! I'm reaching out because our mutual friend recommended you. I have a great business opportunity.", 1, "friendly"),
    ("I know this might seem strange, but I really need someone trustworthy to help me with a sensitive matter.", 1, "friendly"),
    ("We've been helping people like you for 10 years. We just want to make sure you're protected.", 1, "friendly"),
    ("I found your profile and I think you'd be perfect for this. I hope we can build a friendship first.", 1, "friendly"),
    ("Don't tell anyone about this offer. It's only for a select few people I personally chose.", 1, "friendly"),

    # Phishing
    ("Dear customer, your Netflix subscription has expired. Update your payment method: http://netfIix-secure.com", 1, "urgent"),
    ("Your Amazon order has been placed. If this wasn't you, click here to cancel: http://amazoon-verify.net", 1, "urgent"),
    ("Chase Bank: Unusual activity detected. Secure your account at http://chase-secure-login.com/verify", 1, "urgent"),
    ("Your Apple ID has been locked. Unlock it now at: http://apple-id-verify-secure.com/unlock", 1, "urgent"),
    ("IRS Tax Refund: You are owed $3,240. Provide your direct deposit information to receive it.", 1, "greedy"),
    ("Congratulations! As a valued customer you qualify for a free cruise vacation. Call to claim!", 1, "greedy"),
]

LEGITIMATE_MESSAGES = [
    ("Hi! Can we reschedule our meeting to 3pm tomorrow? Let me know if that works.", 0, "neutral"),
    ("Your order #48291 has shipped and will arrive by Thursday. Track it on our website.", 0, "neutral"),
    ("Thanks for applying. We've reviewed your resume and would like to schedule an interview.", 0, "neutral"),
    ("Reminder: Your dentist appointment is on Friday at 10am. Reply to confirm or cancel.", 0, "neutral"),
    ("Hey, are you free this weekend? We're having a small get-together Saturday evening.", 0, "neutral"),
    ("Your monthly statement is ready to view online. As usual, no action is needed.", 0, "neutral"),
    ("The software update you requested has been completed. Please restart to apply changes.", 0, "neutral"),
    ("Hi, I just wanted to check in and see how you're doing. Hope everything is well!", 0, "neutral"),
    ("Please find attached the quarterly report we discussed in this morning's meeting.", 0, "neutral"),
    ("Your refund of $42.50 has been processed and will appear in 3-5 business days.", 0, "neutral"),
    ("Can you review the attached document before the end of day? Thanks in advance.", 0, "neutral"),
    ("Your library book is due back by next Monday. You can renew online to avoid a fine.", 0, "neutral"),
    ("Welcome to our newsletter! You can unsubscribe at any time from the footer link.", 0, "neutral"),
    ("The team is doing great work on the project. Keep it up, and let's sync next week.", 0, "neutral"),
    ("I'm following up on our conversation last week about the proposal. Any update?", 0, "neutral"),
    ("Your package was delivered to the front door at 2:14 PM. Photo confirmation attached.", 0, "neutral"),
    ("Just a heads-up, the office will be closed on Monday for the public holiday.", 0, "neutral"),
    ("Your 2-factor authentication code is 847291. This code expires in 10 minutes.", 0, "neutral"),
    ("We've received your support ticket #10284. Our team will get back to you within 24 hours.", 0, "neutral"),
    ("Congrats on completing the course! Your certificate has been sent to your email.", 0, "neutral"),
    ("The meeting notes from Tuesday's session have been shared in the project folder.", 0, "neutral"),
    ("New comment on your post: 'Great analysis! Really appreciated the detailed breakdown.'", 0, "neutral"),
    ("Your payment of $120.00 was received successfully. Receipt ID: TXN-2024-88219.", 0, "neutral"),
    ("Looking forward to seeing you at the conference next month! Let's connect there.", 0, "neutral"),
    ("Your subscription renews automatically on the 15th. Manage it in account settings.", 0, "neutral"),
    ("I finished reviewing the codebase — left some comments in the pull request.", 0, "neutral"),
    ("The weather looks great this weekend. Want to plan that hike we talked about?", 0, "neutral"),
    ("Our records show your warranty is still valid. Contact us if you need any support.", 0, "neutral"),
    ("Your friend Alice shared a photo album with you. Click to view it.", 0, "neutral"),
    ("We wanted to thank you for your continued support. Here's 10% off your next order.", 0, "neutral"),
]


def build_dataset(extra_noise: float = 0.0) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: [text, label, sentiment_label].

    Parameters
    ----------
    extra_noise : float
        If > 0, randomly flip this fraction of labels to simulate noisy data.
    """
    all_data = SCAM_MESSAGES + LEGITIMATE_MESSAGES
    random.shuffle(all_data)

    df = pd.DataFrame(all_data, columns=["text", "label", "sentiment_label"])

    if extra_noise > 0:
        n_flip = int(len(df) * extra_noise)
        flip_idx = random.sample(range(len(df)), n_flip)
        df.loc[flip_idx, "label"] = 1 - df.loc[flip_idx, "label"]

    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.groupby(["label", "sentiment_label"]).size())
    df.to_csv("sample_data.csv", index=False)
    print("Saved sample_data.csv")
