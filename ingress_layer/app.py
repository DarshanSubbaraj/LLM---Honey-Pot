import logging
from datetime import datetime
import random
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"

# 🧠 IN-MEMORY STATE SYSTEM (Simulating Sequence Modeling & Persistence)
# Tracks the rolling score, message order, and escalation signals across turns
CONVERSATION_MEMORY = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    CONVERSATION_MEMORY[user_id] = {"score": 0.0, "turns": 0}
    await update.message.reply_text("Hello? Who is this? Did someone give you my number?")

async def handle_incoming_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    user_id = message.from_user.id
    incoming_text = message.text.lower()

    # Initialize memory if the scammer bypassed /start
    if user_id not in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[user_id] = {"score": 0.0, "turns": 0}

    session = CONVERSATION_MEMORY[user_id]
    session["turns"] += 1

    # 📊 1. SCAM SIGNAL ENGINE: Extracting Multi-dimensional Signals
    keyword_signal = any(w in incoming_text for w in ["upi", "card", "bank", "money", "pay", "id", "number", "otp", "account"])
    semantic_signal = any(w in incoming_text for w in ["scandal", "hurry", "fast", "now", "emergency", "police", "lock", "blocked", "illegal"])
    behavioral_signal = len(incoming_text) < 4 or incoming_text.endswith("?") # Rapid pacing / demanding checking

    # 📈 2. ROLLING PROBABILITY CONSTANT (P_scam)
    # Suspicions accumulate naturally across turns instead of a binary switch
    weight = 0.0
    if keyword_signal: weight += 0.25
    if semantic_signal: weight += 0.35
    if behavioral_signal: weight += 0.15
    
    # Update rolling probability score (cap at 1.0)
    session["score"] = min(1.0, session["score"] + weight)
    p_scam = session["score"]

    # Silent Logging to YOUR Terminal for Evidence Construction
    print(f"\n🍯 [INGRESS PIPELINE] Turn {session['turns']} Metrics Captured:")
    print(f"👤 Sender: @{message.from_user.username} | ID: {user_id}")
    print(f"💬 Text:   \"{message.text}\"")
    print(f"📈 Current Scam Probability (P_scam): {p_scam:.2f}")
    print("-" * 50)

    # 🎛️ 3. DEFERRED ACTIVATION DEFAULTS (Decision Gate Routing)
    
    # STAGE A: OBSERVE (P_scam < 0.30) - Keep it completely cool and normal
    if p_scam < 0.30:
        bot_reply = random.choice([
            "Hello? Yes, I'm here. Who is this?",
            "Hi, sorry I was a bit busy. What's this regarding?",
            "Yes? Can I help you with something?"
        ])

    # STAGE B: SUSPECT (0.30 <= P_scam < 0.70) - Mild confusion, subtle stalling
    elif 0.30 <= p_scam < 0.70:
        bot_reply = random.choice([
            "Wait, what do you mean by that? I don't really understand...",
            "Hold on, my phone is charging in the kitchen, let me run and grab it!",
            "Is there a problem? Let me check my bank app, but the internet is super slow right now."
        ])

    # STAGE C: ENGAGE (P_scam >= 0.70) - Fully unleash the panicked Honey Trap Persona
    else:
        bot_reply = random.choice([
            "Oh my god, wait, what scandal?! Please don't lock my account, my parents will find out! 😨",
            "Please don't do anything drastic! I'm trying to find my UPI ID right now, give me an exact minute!",
            "I'm getting really scared, my hands are literally shaking. Can you walk me through what to do step-by-step?",
            "The app keeps freezing on me! Should I switch off my Wi-Fi? Please don't close it yet!!"
        ])

    await message.reply_text(bot_reply)

def main():
    print("⚡ Stateful Honey-Trap Ingress Pipeline Online. Monitoring sequence stories...")
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_incoming_message))
    application.run_polling()

if __name__ == '__main__':
    main()