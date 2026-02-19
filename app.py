"""
TechSupport Hub Bot — Advanced Edition
Run:  pip install scikit-learn numpy flask flask-cors
      python app.py
Then open  http://localhost:5000

Advanced features:
  - Per-session context memory (remembers what was discussed)
  - Multi-intent detection (handles two topics in one message)
  - Frustration detection with empathetic prefix
  - Repeat-issue detection → escalation nudge
  - Related-topic suggestions shown as clickable chips after each reply
  - Escalation path: connects user to support team
  - Typo + contraction normalization
  - No scores or debug info shown to users — only clean responses
"""

import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_DATA = [
    # GREETING
    ("hello", "greeting"), ("hi", "greeting"), ("hey", "greeting"),
    ("hi there", "greeting"), ("hello there", "greeting"), ("hey there", "greeting"),
    ("good morning", "greeting"), ("good afternoon", "greeting"), ("good evening", "greeting"),
    ("greetings", "greeting"), ("yo", "greeting"), ("sup", "greeting"),
    ("whats up", "greeting"), ("howdy", "greeting"), ("hiya", "greeting"),
    ("morning", "greeting"), ("evening", "greeting"), ("afternoon", "greeting"),
    ("hi bot", "greeting"), ("hello bot", "greeting"), ("hey can you help", "greeting"),
    ("hi support", "greeting"), ("hello support", "greeting"), ("good day", "greeting"),
    ("hey support team", "greeting"), ("hi is anyone there", "greeting"),
    ("hello is this support", "greeting"), ("hi i need help", "greeting"),
    ("hello i need assistance", "greeting"),

    # FAREWELL
    ("goodbye", "farewell"), ("bye", "farewell"), ("bye bye", "farewell"),
    ("see you", "farewell"), ("see ya", "farewell"), ("thanks bye", "farewell"),
    ("thank you goodbye", "farewell"), ("ok bye", "farewell"), ("alright bye", "farewell"),
    ("have a good day", "farewell"), ("have a nice day", "farewell"), ("take care", "farewell"),
    ("catch you later", "farewell"), ("later", "farewell"), ("im done", "farewell"),
    ("thats all", "farewell"), ("nothing else", "farewell"), ("no more questions", "farewell"),
    ("all good now", "farewell"), ("problem solved thanks", "farewell"),
    ("that helped thanks bye", "farewell"), ("appreciate it bye", "farewell"),
    ("ok thanks", "farewell"), ("got it thanks", "farewell"), ("understood thanks", "farewell"),
    ("great thanks", "farewell"), ("perfect thanks", "farewell"), ("cheers", "farewell"),

    # SMALL TALK
    ("how are you", "small_talk"), ("how are you doing", "small_talk"),
    ("how r u", "small_talk"), ("how r you", "small_talk"), ("hows it going", "small_talk"),
    ("what are you", "small_talk"), ("who are you", "small_talk"),
    ("what can you do", "small_talk"), ("tell me about yourself", "small_talk"),
    ("are you a bot", "small_talk"), ("are you a robot", "small_talk"),
    ("are you real", "small_talk"), ("are you human", "small_talk"),
    ("whats your name", "small_talk"), ("what is your name", "small_talk"),
    ("you doing ok", "small_talk"), ("hows your day", "small_talk"),
    ("nice to meet you", "small_talk"), ("pleasure to meet you", "small_talk"),

    # ACCOUNT ACCESS
    ("cant log in", "account_access"), ("can't login", "account_access"),
    ("cannot log in", "account_access"), ("unable to login", "account_access"),
    ("login not working", "account_access"), ("login problem", "account_access"),
    ("login issue", "account_access"), ("cant sign in", "account_access"),
    ("cannot sign in", "account_access"), ("unable to sign in", "account_access"),
    ("forgot password", "account_access"), ("forgot my password", "account_access"),
    ("dont remember password", "account_access"), ("lost password", "account_access"),
    ("need to reset password", "account_access"), ("reset password", "account_access"),
    ("password reset", "account_access"), ("how to reset password", "account_access"),
    ("forgot username", "account_access"), ("forgot my username", "account_access"),
    ("account locked", "account_access"), ("my account is locked", "account_access"),
    ("locked out of account", "account_access"), ("locked out", "account_access"),
    ("account suspended", "account_access"), ("account disabled", "account_access"),
    ("account blocked", "account_access"), ("access denied", "account_access"),
    ("login failed", "account_access"), ("login keeps failing", "account_access"),
    ("wrong password error", "account_access"), ("invalid credentials", "account_access"),
    ("invalid password", "account_access"), ("incorrect password", "account_access"),
    ("password not accepted", "account_access"), ("password doesnt work", "account_access"),
    ("password not working", "account_access"), ("getting login error", "account_access"),
    ("cant access my account", "account_access"), ("cannot access account", "account_access"),
    ("authentication failed", "account_access"), ("authentication error", "account_access"),
    ("two factor not working", "account_access"), ("2fa problem", "account_access"),
    ("2fa not working", "account_access"), ("verification code not working", "account_access"),
    ("didnt receive verification code", "account_access"),
    ("verification code expired", "account_access"),
    ("too many login attempts", "account_access"), ("account temporarily locked", "account_access"),
    ("need to unlock account", "account_access"), ("email not recognized", "account_access"),
    ("username not found", "account_access"), ("account not found", "account_access"),
    ("session expired", "account_access"), ("keeps logging me out", "account_access"),
    ("automatically logged out", "account_access"), ("login keeps timing out", "account_access"),
    ("i cant login", "account_access"), ("i cant get in", "account_access"),
    ("wont let me login", "account_access"), ("my login aint working", "account_access"),
    ("pasword not working", "account_access"), ("passwrd reset", "account_access"),
    ("logn problem", "account_access"), ("acount locked", "account_access"),
    ("cant acess my account", "account_access"),

    # TECHNICAL ISSUES
    ("app crashing", "technical_issue"), ("app crashes", "technical_issue"),
    ("application crashing", "technical_issue"), ("keeps crashing", "technical_issue"),
    ("crashes when i open it", "technical_issue"), ("crashes on startup", "technical_issue"),
    ("crashes randomly", "technical_issue"), ("app freezes", "technical_issue"),
    ("app freezing", "technical_issue"), ("keeps freezing", "technical_issue"),
    ("frozen screen", "technical_issue"), ("not responding", "technical_issue"),
    ("app not responding", "technical_issue"), ("stuck on loading", "technical_issue"),
    ("loading forever", "technical_issue"), ("wont load", "technical_issue"),
    ("not loading", "technical_issue"), ("error message", "technical_issue"),
    ("getting error", "technical_issue"), ("technical error", "technical_issue"),
    ("system error", "technical_issue"), ("bug in the app", "technical_issue"),
    ("found a bug", "technical_issue"), ("something is broken", "technical_issue"),
    ("not working properly", "technical_issue"), ("app not working", "technical_issue"),
    ("doesnt work", "technical_issue"), ("stopped working", "technical_issue"),
    ("slow performance", "technical_issue"), ("very slow", "technical_issue"),
    ("running slow", "technical_issue"), ("laggy", "technical_issue"),
    ("lagging", "technical_issue"), ("app is slow", "technical_issue"),
    ("taking forever", "technical_issue"), ("not syncing", "technical_issue"),
    ("sync not working", "technical_issue"), ("sync failed", "technical_issue"),
    ("sync error", "technical_issue"), ("connection error", "technical_issue"),
    ("connection failed", "technical_issue"), ("cant connect", "technical_issue"),
    ("cannot connect", "technical_issue"), ("no connection", "technical_issue"),
    ("lost connection", "technical_issue"), ("disconnected", "technical_issue"),
    ("keeps disconnecting", "technical_issue"), ("connection timeout", "technical_issue"),
    ("server error", "technical_issue"), ("server not responding", "technical_issue"),
    ("server down", "technical_issue"), ("service unavailable", "technical_issue"),
    ("500 error", "technical_issue"), ("404 error", "technical_issue"),
    ("network error", "technical_issue"), ("blank screen", "technical_issue"),
    ("white screen", "technical_issue"), ("black screen", "technical_issue"),
    ("nothing shows up", "technical_issue"), ("page not loading", "technical_issue"),
    ("buttons not working", "technical_issue"), ("cant click anything", "technical_issue"),
    ("interface not working", "technical_issue"), ("glitch in the system", "technical_issue"),
    ("glitchy", "technical_issue"), ("display problem", "technical_issue"),
    ("screen flickering", "technical_issue"), ("images not loading", "technical_issue"),
    ("files not uploading", "technical_issue"), ("upload failed", "technical_issue"),
    ("cant upload files", "technical_issue"), ("download not working", "technical_issue"),
    ("cant download", "technical_issue"), ("download failed", "technical_issue"),
    ("data lost", "technical_issue"), ("lost my data", "technical_issue"),
    ("data disappeared", "technical_issue"), ("data missing", "technical_issue"),
    ("corrupted file", "technical_issue"), ("file corrupted", "technical_issue"),
    ("app wont open", "technical_issue"), ("cant open app", "technical_issue"),
    ("wont start", "technical_issue"), ("startup error", "technical_issue"),
    ("installation failed", "technical_issue"), ("cant install", "technical_issue"),
    ("installation error", "technical_issue"), ("update failed", "technical_issue"),
    ("cant update", "technical_issue"), ("update not working", "technical_issue"),
    ("update error", "technical_issue"), ("notification not working", "technical_issue"),
    ("not getting notifications", "technical_issue"),
    ("notifications stopped", "technical_issue"),
    ("search not working", "technical_issue"), ("cant search", "technical_issue"),
    ("search broken", "technical_issue"), ("filter not working", "technical_issue"),
    ("app kep crashing", "technical_issue"), ("appp crashes", "technical_issue"),
    ("aplication not working", "technical_issue"), ("it dosnt work", "technical_issue"),
    ("its not wrking", "technical_issue"), ("the app is brken", "technical_issue"),
    ("i get an eror", "technical_issue"), ("its very slw", "technical_issue"),
    ("wifi not working with app", "technical_issue"), ("cant open the app", "technical_issue"),
    ("the screen is blank", "technical_issue"), ("nothng loads", "technical_issue"),

    # BILLING
    ("billing question", "billing"), ("billing issue", "billing"),
    ("billing problem", "billing"), ("billing help", "billing"),
    ("payment question", "billing"), ("payment issue", "billing"),
    ("payment not working", "billing"), ("payment failed", "billing"),
    ("payment error", "billing"), ("payment declined", "billing"),
    ("card declined", "billing"), ("credit card not working", "billing"),
    ("charge on my card", "billing"), ("unauthorized charge", "billing"),
    ("unexpected charge", "billing"), ("double charge", "billing"),
    ("charged twice", "billing"), ("overcharged", "billing"),
    ("wrong amount charged", "billing"), ("refund", "billing"),
    ("i want a refund", "billing"), ("need a refund", "billing"),
    ("request refund", "billing"), ("how to get refund", "billing"),
    ("money back", "billing"), ("get my money back", "billing"),
    ("cancel subscription", "billing"), ("cancel my subscription", "billing"),
    ("how to cancel", "billing"), ("want to cancel", "billing"),
    ("subscription cancelled", "billing"), ("plan cancelled", "billing"),
    ("how much does it cost", "billing"), ("pricing", "billing"),
    ("what are the plans", "billing"), ("pricing plans", "billing"),
    ("how much is pro", "billing"), ("subscription cost", "billing"),
    ("upgrade plan", "billing"), ("downgrade plan", "billing"),
    ("change plan", "billing"), ("invoice", "billing"),
    ("need invoice", "billing"), ("billing receipt", "billing"),
    ("tax invoice", "billing"), ("payment receipt", "billing"),
    ("update payment method", "billing"), ("change credit card", "billing"),
    ("add new card", "billing"), ("promo code", "billing"),
    ("discount code", "billing"), ("coupon", "billing"),
    ("free trial", "billing"), ("trial expired", "billing"),
    ("subscription expired", "billing"), ("payment keeps failing", "billing"),
    ("cant pay", "billing"), ("student discount", "billing"),
    ("nonprofit discount", "billing"), ("annual plan", "billing"),
    ("monthly plan", "billing"), ("billing cycle", "billing"),
    ("when am i charged", "billing"), ("next billing date", "billing"),
    ("how mch does it cost", "billing"), ("pricng plans", "billing"),
    ("i want refund", "billing"), ("canel subscription", "billing"),

    # FEATURE HELP
    ("how do i export data", "feature_help"), ("export my data", "feature_help"),
    ("download my data", "feature_help"), ("share with team", "feature_help"),
    ("how to share", "feature_help"), ("invite team member", "feature_help"),
    ("create backup", "feature_help"), ("how to backup", "feature_help"),
    ("how to sync", "feature_help"), ("import file", "feature_help"),
    ("how to import", "feature_help"), ("use api", "feature_help"),
    ("api documentation", "feature_help"), ("keyboard shortcuts", "feature_help"),
    ("mobile app", "feature_help"), ("use on mobile", "feature_help"),
    ("how to use", "feature_help"), ("tutorial", "feature_help"),
    ("help with feature", "feature_help"), ("guide me", "feature_help"),
    ("step by step", "feature_help"), ("how does this work", "feature_help"),
    ("how to set up", "feature_help"), ("configure settings", "feature_help"),
    ("customize", "feature_help"), ("change settings", "feature_help"),
    ("where is the setting", "feature_help"), ("cant find feature", "feature_help"),
    ("where do i", "feature_help"), ("how to enable", "feature_help"),
    ("how to disable", "feature_help"), ("how do i use", "feature_help"),

    # ACCOUNT MANAGEMENT
    ("delete my account", "account_management"), ("how to delete account", "account_management"),
    ("close my account", "account_management"), ("remove my account", "account_management"),
    ("change my email", "account_management"), ("update email address", "account_management"),
    ("change email", "account_management"), ("update profile", "account_management"),
    ("edit my profile", "account_management"), ("change my name", "account_management"),
    ("change password", "account_management"), ("update password", "account_management"),
    ("enable 2fa", "account_management"), ("setup two factor", "account_management"),
    ("disable 2fa", "account_management"), ("turn off 2fa", "account_management"),
    ("privacy settings", "account_management"), ("notification settings", "account_management"),
    ("language settings", "account_management"), ("change language", "account_management"),
    ("change timezone", "account_management"), ("account settings", "account_management"),
    ("profile settings", "account_management"), ("user settings", "account_management"),
    ("manage account", "account_management"), ("deactivate account", "account_management"),

    # DATA / PRIVACY
    ("export all my data", "data_export"), ("download all data", "data_export"),
    ("data export", "data_export"), ("privacy policy", "data_export"),
    ("who can see my data", "data_export"), ("data security", "data_export"),
    ("is my data safe", "data_export"), ("gdpr request", "data_export"),
    ("data deletion request", "data_export"), ("delete my data", "data_export"),
    ("remove my data", "data_export"), ("data privacy", "data_export"),
    ("data breach", "data_export"), ("my data was leaked", "data_export"),
    ("what data do you collect", "data_export"), ("data policy", "data_export"),

    # GENERAL INQUIRY
    ("what features do you have", "general_inquiry"),
    ("tell me about the product", "general_inquiry"),
    ("what is this product", "general_inquiry"), ("about the app", "general_inquiry"),
    ("system requirements", "general_inquiry"), ("what platforms", "general_inquiry"),
    ("is there a mobile app", "general_inquiry"), ("browser support", "general_inquiry"),
    ("does it work on mac", "general_inquiry"), ("does it work on windows", "general_inquiry"),
    ("offline mode", "general_inquiry"), ("integration", "general_inquiry"),
    ("does it integrate with", "general_inquiry"), ("compatibility", "general_inquiry"),
    ("supported languages", "general_inquiry"), ("available in my country", "general_inquiry"),
    ("team size", "general_inquiry"), ("how many users", "general_inquiry"),
    ("enterprise plan", "general_inquiry"),

    # POSITIVE FEEDBACK
    ("this is amazing", "feedback_positive"), ("i love it", "feedback_positive"),
    ("great product", "feedback_positive"), ("works perfectly", "feedback_positive"),
    ("thank you so much", "feedback_positive"), ("you guys are great", "feedback_positive"),
    ("excellent service", "feedback_positive"), ("very helpful", "feedback_positive"),
    ("problem is solved", "feedback_positive"), ("it worked", "feedback_positive"),
    ("issue resolved", "feedback_positive"), ("everything is working now", "feedback_positive"),
    ("fixed now thank you", "feedback_positive"), ("that worked thanks", "feedback_positive"),

    # NEGATIVE FEEDBACK
    ("terrible service", "feedback_negative"), ("this sucks", "feedback_negative"),
    ("very disappointed", "feedback_negative"), ("worst app ever", "feedback_negative"),
    ("not happy with service", "feedback_negative"), ("really frustrated", "feedback_negative"),
    ("this is ridiculous", "feedback_negative"), ("awful experience", "feedback_negative"),
    ("unhappy customer", "feedback_negative"), ("waste of money", "feedback_negative"),

    # ESCALATION
    ("talk to a human", "escalation"), ("speak to an agent", "escalation"),
    ("connect me to support", "escalation"), ("i want real support", "escalation"),
    ("talk to a person", "escalation"), ("human agent please", "escalation"),
    ("this bot is useless", "escalation"), ("let me talk to someone", "escalation"),
    ("i need real help", "escalation"), ("speak to a representative", "escalation"),
    ("contact support team", "escalation"), ("i need to email support", "escalation"),
    ("live chat", "escalation"), ("phone support", "escalation"),
    ("raise a ticket", "escalation"), ("open a ticket", "escalation"),
    ("create support ticket", "escalation"), ("submit a request", "escalation"),

    # OUT OF SCOPE
    ("need custom development", "out_of_scope"), ("build me a website", "out_of_scope"),
    ("custom software", "out_of_scope"), ("business partnership", "out_of_scope"),
    ("partnership opportunity", "out_of_scope"), ("job application", "out_of_scope"),
    ("looking for work", "out_of_scope"), ("are you hiring", "out_of_scope"),
    ("investment opportunity", "out_of_scope"), ("media inquiry", "out_of_scope"),
    ("press release", "out_of_scope"), ("marketing question", "out_of_scope"),
    ("legal question", "out_of_scope"), ("legal advice", "out_of_scope"),
    ("sue your company", "out_of_scope"), ("lawyer", "out_of_scope"),
    ("whats the weather", "out_of_scope"), ("tell me a joke", "out_of_scope"),
    ("what is the capital of france", "out_of_scope"), ("write me a poem", "out_of_scope"),
    ("what is 2 plus 2", "out_of_scope"), ("help me with homework", "out_of_scope"),
    ("stock price", "out_of_scope"), ("news today", "out_of_scope"),
    ("recommend a restaurant", "out_of_scope"), ("translate this", "out_of_scope"),
]

# ─────────────────────────────────────────────────────────────────────────────
#  RESPONSES
# ─────────────────────────────────────────────────────────────────────────────

RESPONSES = {
    "greeting": (
        "Hey there! Welcome to TechSupport Hub. 😊\n"
        "I'm here to help with account issues, technical problems, billing, features, and more.\n\n"
        "What can I help you with today?"
    ),
    "farewell": (
        "Glad I could help! Take care and have a great day. 👋\n"
        "If you run into anything else, I'm always here."
    ),
    "small_talk": (
        "I'm your TechSupport Hub assistant — doing great and ready to help! 🤖\n\n"
        "I specialize in account issues, technical problems, billing, and product features. "
        "What can I sort out for you today?"
    ),
    "account_access": (
        "I can help you get back into your account! Let's work through this.\n\n"
        "**🔑 Password issues:**\n"
        "1. Go to **techsupporthub.com/login**\n"
        "2. Click **'Forgot Password'** below the login button\n"
        "3. Enter your registered email address\n"
        "4. Check your inbox — and your **spam/junk folder** too\n"
        "5. The reset link expires in **15 minutes** — use it promptly\n\n"
        "**🔒 Account locked / too many attempts:**\n"
        "• Accounts auto-unlock after **30 minutes**\n"
        "• For immediate unlock: email **support@techsupporthub.com**\n\n"
        "**📱 2FA / verification code issues:**\n"
        "• Make sure your **device time is synced** (2FA is time-sensitive)\n"
        "• Codes refresh every 30 seconds — wait for a new one\n"
        "• Backup option: request the code via **SMS** instead\n\n"
        "**💡 Quick checklist:**\n"
        "• Try a **different browser** or incognito/private mode\n"
        "• Clear your **browser cookies and cache**\n"
        "• Make sure Caps Lock is off\n\n"
        "Did any of these steps help? If not, just let me know and I'll connect you to our team."
    ),
    "technical_issue": (
        "Let's get that sorted — most issues are fixed in a few steps.\n\n"
        "**⚡ Try these first (fixes 80% of issues):**\n"
        "1. **Force close & reopen** the app completely\n"
        "2. **Check your internet** — switch between Wi-Fi and mobile data to test\n"
        "3. **Clear cache:**\n"
        "   • Android: Settings → Apps → TechSupport Hub → Storage → Clear Cache\n"
        "   • iPhone: Delete & reinstall (your data is cloud-saved)\n"
        "   • Browser: Ctrl+Shift+Delete → clear cache & cookies\n"
        "4. **Update the app** — outdated versions cause most crash/freeze bugs\n\n"
        "**📱 Device-specific fixes:**\n"
        "• **iPhone crash:** Swipe up → swipe app away → reopen\n"
        "• **Android crash:** Settings → Apps → TechSupport Hub → Force Stop → reopen\n"
        "• **Desktop:** Fully quit (check system tray) and restart\n\n"
        "**🐌 Slow or laggy?**\n"
        "• Close background apps to free up RAM\n"
        "• Check if device storage is above 90% full\n"
        "• Restart your device\n\n"
        "**🔄 Sync or connection errors?**\n"
        "• Log out and log back in\n"
        "• Toggle airplane mode on/off to reset the connection\n\n"
        "Still happening? Tell me what error message you're seeing and I'll dig deeper."
    ),
    "billing": (
        "Happy to help with billing!\n\n"
        "**💰 Our Plans:**\n"
        "• **Free** — $0/month | 1GB storage | Community support\n"
        "• **Pro** — $9.99/month | 100GB | Priority email support | 30-day free trial\n"
        "• **Team** — $29.99/month | 500GB | Up to 10 members | Priority phone support\n\n"
        "**🎁 Discounts:**\n"
        "• Students **50% off** (verify with .edu email)\n"
        "• Nonprofits **40% off** (email billing to apply)\n"
        "• Annual billing **20% off** (2 months free)\n\n"
        "**💳 Payment failed or card declined:**\n"
        "• Settings → Billing → **Update Payment Method**\n"
        "• Make sure billing address matches your card exactly\n"
        "• Try PayPal as an alternative\n\n"
        "**🔄 Cancel subscription:**\n"
        "• Settings → Subscription → **Cancel Plan**\n"
        "• You keep access until the end of your paid period — no cancellation fee\n\n"
        "**💵 Refund requests:**\n"
        "• Email **billing@techsupporthub.com** within **30 days** of charge\n"
        "• Include your account email and reason\n"
        "• Refunds processed in **5–7 business days**\n\n"
        "**📄 Need an invoice?** Email billing@techsupporthub.com — sent within 24 hours.\n\n"
        "Anything specific about billing I can help clarify?"
    ),
    "feature_help": (
        "Happy to walk you through it!\n\n"
        "**📤 Exporting your data:**\n"
        "• Settings → Privacy → **Export My Data**\n"
        "• Choose format: CSV, JSON, or PDF\n"
        "• Download link arrives in your email within **24 hours**\n\n"
        "**👥 Inviting team members:**\n"
        "• Settings → Team → **Invite Members**\n"
        "• Enter email → choose role: Viewer, Editor, or Admin\n"
        "• Invite link is valid for **7 days**\n\n"
        "**🔄 Sync & backup:**\n"
        "• Auto-sync every **5 minutes** when online\n"
        "• Force sync: pull down on home screen or Settings → **Sync Now**\n"
        "• Manual backup: Settings → Storage → **Create Backup**\n\n"
        "**📱 Mobile app:**\n"
        "• iOS: Search 'TechSupport Hub' on **App Store**\n"
        "• Android: Search 'TechSupport Hub' on **Google Play**\n"
        "• Data syncs automatically after login\n\n"
        "**⌨️ Keyboard shortcuts (desktop):**\n"
        "• Ctrl+S: Save • Ctrl+Z: Undo • Ctrl+/: Command palette\n\n"
        "**📖 Full documentation:** techsupporthub.com/help\n\n"
        "Which feature are you trying to use? I can give more specific steps!"
    ),
    "account_management": (
        "Here's how to update your account:\n\n"
        "**✏️ Change name or profile info:**\n"
        "• Profile icon (top right) → **Edit Profile** → Save\n\n"
        "**📧 Change email address:**\n"
        "• Settings → Account → **Change Email**\n"
        "• A verification link is sent to your new email\n"
        "• Old email stays active until you verify the new one\n\n"
        "**🔑 Change password:**\n"
        "• Settings → Security → **Change Password**\n"
        "• You need your current password to set a new one\n\n"
        "**🛡️ Two-Factor Authentication (2FA):**\n"
        "• Enable: Settings → Security → 2FA → **Enable** → scan QR with Authenticator app\n"
        "• Disable: Settings → Security → 2FA → **Disable** (requires password confirmation)\n\n"
        "**🔔 Notification preferences:**\n"
        "• Settings → Notifications → toggle what you want on/off\n\n"
        "**⚠️ Delete or deactivate account:**\n"
        "• **Deactivate** (reversible): Settings → Account → Deactivate — data preserved\n"
        "• **Delete** (permanent): Settings → Account → Delete Account — cannot be undone\n"
        "• Before deleting: export your data first!\n\n"
        "Which setting are you trying to change? I can guide you exactly."
    ),
    "data_export": (
        "Your data privacy is our top priority. Here's everything you need:\n\n"
        "**📥 Download a copy of your data:**\n"
        "• Settings → Privacy → **Export My Data**\n"
        "• Select date range + format (CSV or JSON)\n"
        "• Secure download link sent to your email within **24 hours**\n"
        "• Link valid for **72 hours** — download promptly\n\n"
        "**🔒 How we protect your data:**\n"
        "• **In transit:** SSL/TLS encryption on all connections\n"
        "• **At rest:** AES-256 encryption in our databases\n"
        "• We **never sell** your data to third parties\n"
        "• Full privacy policy: **techsupporthub.com/privacy**\n\n"
        "**🇪🇺 GDPR / right to be forgotten:**\n"
        "• Email **privacy@techsupporthub.com** with subject: 'GDPR Request'\n"
        "• Processed within **30 days** as required by law\n\n"
        "**🚨 Suspected breach or unauthorized access:**\n"
        "• Email **security@techsupporthub.com** immediately\n"
        "• Change your password and revoke active sessions right away\n"
        "• Our security team responds within **1 hour**, monitors 24/7\n\n"
        "Do you have a specific privacy concern I can help address?"
    ),
    "general_inquiry": (
        "Great question! Here's an overview of TechSupport Hub:\n\n"
        "**📱 Supported platforms:**\n"
        "• Mobile: iOS 14+ and Android 8+\n"
        "• Desktop: Windows 10+, macOS 11+, Linux (Ubuntu/Debian)\n"
        "• Browser: Chrome, Firefox, Safari, Edge (latest 2 versions)\n\n"
        "**🔌 Integrations:**\n"
        "• Slack, Google Drive, Dropbox, OneDrive, Zapier\n"
        "• REST API available on Pro and Team plans\n"
        "• SSO via Google, Microsoft, or SAML 2.0 (Team plan)\n\n"
        "**🌍 Availability:**\n"
        "• Available in **25+ languages** across **150+ countries**\n"
        "• Offline mode available on mobile\n\n"
        "**📖 Full features:** techsupporthub.com/features\n"
        "**🧑‍💼 Enterprise pricing:** enterprise@techsupporthub.com\n\n"
        "Is there something specific you'd like to know more about?"
    ),
    "feedback_positive": (
        "That's wonderful to hear — thank you so much! 🎉\n\n"
        "It genuinely means a lot. If you ever need help with anything, "
        "just come back and I'm here. Have a great day!"
    ),
    "feedback_negative": (
        "I'm truly sorry you've had a frustrating experience — that's not what we want at all.\n\n"
        "I really want to help make this right. Could you tell me:\n"
        "• **What specifically went wrong?**\n"
        "• **How long has this been happening?**\n\n"
        "The more detail you share, the better I can help — and if I can't solve it, "
        "I'll connect you directly to our support team who will prioritize your case."
    ),
    "escalation": (
        "Absolutely, let me connect you with our support team.\n\n"
        "**📧 Email support (fastest response):**\n"
        "• General issues: **support@techsupporthub.com**\n"
        "• Billing issues: **billing@techsupporthub.com**\n"
        "• Technical issues: **tech@techsupporthub.com**\n"
        "• Response time: **within 2 hours** (Mon–Fri, 9am–6pm EST)\n\n"
        "**🎫 Submit a support ticket:**\n"
        "• Go to **techsupporthub.com/support** → click 'New Ticket'\n"
        "• You'll receive a ticket number and email updates on progress\n\n"
        "**💬 Live chat:**\n"
        "• Available for **Pro and Team** plan members\n"
        "• Look for the chat bubble on **techsupporthub.com**\n\n"
        "**📞 Phone support:**\n"
        "• Available for **Team plan** members\n"
        "• Number is in your account dashboard under Support\n\n"
        "Before I hand you off — would you like to quickly describe your issue? "
        "I might still be able to help you right now!"
    ),
    "out_of_scope": (
        "I appreciate you reaching out! That's outside my area — "
        "I'm focused on TechSupport Hub product support.\n\n"
        "**What I can help you with:**\n"
        "• 🔐 Login and account access\n"
        "• 🛠️ App errors, crashes, and performance\n"
        "• 💳 Billing, subscriptions, and refunds\n"
        "• 📖 How to use product features\n"
        "• ⚙️ Account settings and data privacy\n\n"
        "For anything else, reach out at **support@techsupporthub.com** and the right team will help. 😊"
    ),
}

LOW_CONFIDENCE_RESPONSE = (
    "Hmm, I want to make sure I give you the right answer — could you give me a bit more detail?\n\n"
    "For example:\n"
    "• *'I can't log into my account'*\n"
    "• *'The app keeps crashing on my phone'*\n"
    "• *'I want to cancel my subscription'*\n"
    "• *'How do I export my data?'*\n\n"
    "I'm here and happy to help once I understand what you need!"
)

RELATED_TOPICS = {
    "account_access":     ["account_management", "technical_issue"],
    "technical_issue":    ["account_access", "feature_help"],
    "billing":            ["account_management", "general_inquiry"],
    "feature_help":       ["technical_issue", "data_export"],
    "account_management": ["account_access", "data_export"],
    "data_export":        ["account_management", "billing"],
    "general_inquiry":    ["billing", "feature_help"],
    "feedback_negative":  ["technical_issue", "billing"],
    "escalation":         ["technical_issue", "billing"],
}

TOPIC_LABELS = {
    "account_access":     "Login & account access",
    "technical_issue":    "Technical problems",
    "billing":            "Billing & subscriptions",
    "feature_help":       "Using features",
    "account_management": "Account settings",
    "data_export":        "Data & privacy",
    "general_inquiry":    "About the product",
    "escalation":         "Contact support team",
}

SUGGESTION_QUERIES = {
    "account_access":     [("🔧 Technical issues", "my app is not working"), ("⚙️ Account settings", "how to change my password")],
    "technical_issue":    [("🔐 Login issues", "i cant login"), ("📖 Feature help", "how to use the app")],
    "billing":            [("⚙️ Account settings", "how to manage my account"), ("❓ About the product", "what features do you have")],
    "feature_help":       [("🛠️ Technical issues", "something is broken"), ("📤 Export data", "how to export my data")],
    "account_management": [("🔐 Login & access", "i cant access my account"), ("🔒 Data & privacy", "is my data safe")],
    "data_export":        [("⚙️ Account settings", "delete my account"), ("💳 Billing", "cancel my subscription")],
    "general_inquiry":    [("💰 See pricing", "how much does it cost"), ("📖 Feature help", "how to use the features")],
    "feedback_negative":  [("🛠️ Technical help", "app not working"), ("💳 Billing help", "i want a refund")],
    "escalation":         [("🛠️ Technical help", "app crashing"), ("💳 Billing help", "payment issue")],
}

# ─────────────────────────────────────────────────────────────────────────────
#  FRUSTRATION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

FRUSTRATION_WORDS = {
    "frustrated", "angry", "annoyed", "upset", "furious", "ridiculous",
    "useless", "terrible", "horrible", "awful", "pathetic", "disgusting",
    "hate", "worst", "garbage", "trash", "stupid", "incompetent",
    "waste", "scam", "fraud", "ripped off", "unacceptable",
    "unbelievable", "outrageous", "fed up", "sick of", "done with",
    "never works", "keeps happening", "same issue", "same problem",
    "still broken", "still not working", "days", "weeks",
}

def detect_frustration(text: str) -> bool:
    words = set(text.lower().split())
    return bool(words & FRUSTRATION_WORDS)

def frustration_prefix() -> str:
    return (
        "I can hear how frustrated you are, and I'm genuinely sorry — you deserve better than this. "
        "Let me do everything I can to help you right now.\n\n"
    )

# ─────────────────────────────────────────────────────────────────────────────
#  TEXT NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────

CONTRACTIONS = {
    "can't": "cannot", "cant": "cannot", "won't": "will not", "wont": "will not",
    "don't": "do not", "dont": "do not", "didn't": "did not", "didnt": "did not",
    "doesn't": "does not", "doesnt": "does not", "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not", "wasn't": "was not", "wasnt": "was not",
    "haven't": "have not", "havent": "have not", "hasn't": "has not", "hasnt": "has not",
    "i'm": "i am", "im": "i am", "you're": "you are", "youre": "you are",
    "it's": "it is", "we're": "we are", "they're": "they are", "theyre": "they are",
    "i've": "i have", "ive": "i have", "you've": "you have", "youve": "you have",
    "i'll": "i will", "ill": "i will", "you'll": "you will", "youll": "you will",
    "i'd": "i would", "wanna": "want to", "gonna": "going to", "gotta": "got to",
    "kinda": "kind of", "plz": "please", "pls": "please",
    "thx": "thanks", "ty": "thank you", "bc": "because", "cuz": "because",
    "r": "are", "u": "you", "ur": "your", "aint": "am not",
    "omg": "oh my", "wtf": "what the",
}

TYPO_MAP = {
    "pasword": "password", "passwrd": "password", "passwd": "password",
    "acount": "account", "accont": "account", "acct": "account",
    "logn": "login", "loogin": "login",
    "technial": "technical", "techincal": "technical",
    "isue": "issue", "isseu": "issue",
    "eror": "error", "errro": "error",
    "billig": "billing", "billin": "billing",
    "subscrition": "subscription", "subscripton": "subscription",
    "cancl": "cancel", "cancle": "cancel", "canel": "cancel", "cancell": "cancel",
    "updaet": "update", "updat": "update",
    "instalation": "installation",
    "aplication": "application", "applciation": "application",
    "sycn": "sync", "snc": "sync",
    "conection": "connection", "connecton": "connection",
    "slowww": "slow", "slw": "slow",
    "freez": "freeze", "frezze": "freeze",
    "pricng": "pricing",
    "acess": "access", "acces": "access",
    "verifcation": "verification",
    "authnetication": "authentication",
    "nothng": "nothing",
    "brken": "broken", "borken": "broken",
    "dosnt": "doesnt", "wrking": "working",
    "appp": "app", "hlep": "help", "hepl": "help",
}

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    words = text.split()
    corrected = []
    for w in words:
        clean_w = re.sub(r"[^a-z0-9']", "", w)
        if clean_w in CONTRACTIONS:
            corrected.append(CONTRACTIONS[clean_w])
        elif clean_w in TYPO_MAP:
            corrected.append(TYPO_MAP[clean_w])
        else:
            corrected.append(w)
    text = " ".join(corrected)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────────────────────────────────────────
#  BOT
# ─────────────────────────────────────────────────────────────────────────────

class TechSupportBot:

    CONFIDENCE_THRESHOLD   = 0.38
    SECONDARY_THRESHOLD    = 0.26

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.pipeline      = None
        self.sessions      = {}
        self._train()

    def _train(self):
        texts  = [normalize_text(t) for t, _ in TRAINING_DATA]
        labels = [label for _, label in TRAINING_DATA]
        y      = self.label_encoder.fit_transform(labels)

        word_vec = TfidfVectorizer(max_features=12000, ngram_range=(1, 2),
                                   min_df=1, max_df=0.95, sublinear_tf=True)
        char_vec = TfidfVectorizer(max_features=12000, ngram_range=(2, 4),
                                   min_df=1, max_df=0.95, sublinear_tf=True,
                                   analyzer="char_wb")
        combined = FeatureUnion([("word", word_vec), ("char", char_vec)])
        clf = LogisticRegression(C=3.0, max_iter=5000,
                                 class_weight="balanced", solver="lbfgs", random_state=42)
        self.pipeline = Pipeline([("features", combined), ("clf", clf)])
        self.pipeline.fit(texts, y)
        print("✅ TechSupport Bot (Advanced) ready!")

    # ── session helpers ──────────────────────────────────────────────────────
    def _ctx(self, sid: str) -> dict:
        if sid not in self.sessions:
            self.sessions[sid] = {
                "history":           [],   # list of (msg, intent, response)
                "discussed":         set(),
                "frustration_count": 0,
                "escalated":         False,
            }
        return self.sessions[sid]

    # ── prediction ───────────────────────────────────────────────────────────
    def _predict_all(self, message: str):
        clean = normalize_text(message)
        probs  = self.pipeline.predict_proba([clean])[0]
        ranked = sorted(zip(self.label_encoder.classes_, probs),
                        key=lambda x: x[1], reverse=True)
        return ranked

    def _top_intents(self, ranked):
        primary   = ranked[0]
        secondary = ranked[1] if len(ranked) > 1 else None
        non_conv  = {"greeting", "farewell", "small_talk", "out_of_scope"}
        if (secondary
                and secondary[1] >= self.SECONDARY_THRESHOLD
                and primary[0]   not in non_conv
                and secondary[0] not in non_conv):
            return primary, secondary
        return primary, None

    # ── main chat ────────────────────────────────────────────────────────────
    def chat(self, message: str, sid: str = "default") -> tuple:
        """Returns (response_text, suggestions_list)."""
        ctx    = self._ctx(sid)
        ranked = self._predict_all(message)
        (intent, confidence), secondary = self._top_intents(ranked)

        is_frustrated = detect_frustration(message)
        if is_frustrated:
            ctx["frustration_count"] += 1

        parts = []

        # ── low confidence ───────────────────────────────────────────────────
        if confidence < self.CONFIDENCE_THRESHOLD:
            # If very short follow-up, assume same topic
            if ctx["history"] and len(message.split()) <= 4:
                intent = ctx["history"][-1][1]
                confidence = self.CONFIDENCE_THRESHOLD
            else:
                ctx["history"].append((message, "unknown", LOW_CONFIDENCE_RESPONSE))
                return LOW_CONFIDENCE_RESPONSE, []

        # ── frustration prefix ───────────────────────────────────────────────
        if is_frustrated and ctx["frustration_count"] <= 2:
            parts.append(frustration_prefix())

        # ── repeated issue → escalation nudge ───────────────────────────────
        is_repeat = (intent in ctx["discussed"]
                     and intent not in ("greeting", "farewell", "small_talk"))
        if is_repeat:
            parts.append(
                "It looks like this issue is still giving you trouble — I'm sorry! "
                "Since the previous steps didn't fully resolve it, the best next move is "
                "to reach our team directly:\n\n"
                "📧 **support@techsupporthub.com** — they respond within 2 hours.\n"
                "Please mention what you've already tried so they can jump straight to a solution.\n\n"
            )
            ctx["escalated"] = True
        else:
            parts.append(RESPONSES.get(intent, LOW_CONFIDENCE_RESPONSE))

        # ── secondary intent block ───────────────────────────────────────────
        if secondary and secondary[0] not in ctx["discussed"]:
            sec_intent, _ = secondary
            sec_label = TOPIC_LABELS.get(sec_intent, sec_intent)
            parts.append(
                f"\n\n---\n**Also, since you mentioned {sec_label}:**\n"
                + RESPONSES.get(sec_intent, "")
            )
            ctx["discussed"].add(sec_intent)

        # ── related-topic hint ───────────────────────────────────────────────
        non_suggest = {"greeting", "farewell", "small_talk", "out_of_scope", "feedback_positive"}
        if intent not in non_suggest:
            related = [t for t in RELATED_TOPICS.get(intent, []) if t not in ctx["discussed"]]
            if related:
                labels = [f"**{TOPIC_LABELS[t]}**" for t in related if t in TOPIC_LABELS]
                if labels:
                    parts.append(
                        f"\n\n💡 *Also commonly asked: {' · '.join(labels)}*"
                    )

        # ── heavy frustration → escalation CTA ──────────────────────────────
        if ctx["frustration_count"] >= 2 and not ctx["escalated"]:
            parts.append(
                "\n\n---\nI really want to make sure this gets fully resolved for you. "
                "Would you like me to help you **connect to our support team**? "
                "Just say *'connect me to support'* and I'll give you everything you need."
            )
            ctx["escalated"] = True

        # ── update context ───────────────────────────────────────────────────
        response = "".join(parts)
        ctx["discussed"].add(intent)
        ctx["history"].append((message, intent, response))

        # build suggestion chips
        suggestions = []
        for label, query in SUGGESTION_QUERIES.get(intent, []):
            suggestions.append({"label": label, "query": query})

        return response, suggestions

    def clear_session(self, sid: str = "default"):
        self.sessions.pop(sid, None)

    def get_history(self, sid: str = "default"):
        ctx = self._ctx(sid)
        return [{"user": h[0], "bot": h[2]} for h in ctx["history"]]

    def predict_all_public(self, message: str):
        return self._predict_all(message)


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

flask_app = Flask(__name__)
CORS(flask_app)

print("\n" + "="*60)
print("  TechSupport Hub Bot (Advanced) — Starting...")
print("="*60)
bot = TechSupportBot()
print("="*60 + "\n")

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TechSupport Hub</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh; display: flex; align-items: center; justify-content: center;
  }
  .chat-wrapper {
    width: 100%; max-width: 820px; height: 94vh;
    display: flex; flex-direction: column;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(24px);
    border-radius: 24px; border: 1px solid rgba(255,255,255,0.12);
    overflow: hidden; box-shadow: 0 32px 80px rgba(0,0,0,0.55);
    margin: 0 16px;
  }
  .header {
    padding: 18px 26px; background: rgba(255,255,255,0.07);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    display: flex; align-items: center; justify-content: space-between;
  }
  .header-left { display: flex; align-items: center; gap: 13px; }
  .header-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #a78bfa, #7c3aed);
    border-radius: 14px; display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
  }
  .header-text h1 { font-size: 17px; font-weight: 700; color: #fff; }
  .header-text p  { font-size: 11px; color: rgba(255,255,255,0.45); margin-top: 2px; }
  .status-dot {
    width: 7px; height: 7px; background: #34d399; border-radius: 50%;
    display: inline-block; margin-right: 5px; animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(52,211,153,0.5); }
    50% { box-shadow: 0 0 0 5px rgba(52,211,153,0); }
  }
  .clear-btn {
    padding: 7px 14px; background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12); border-radius: 10px;
    color: rgba(255,255,255,0.5); font-size: 12px; cursor: pointer; transition: all 0.2s;
  }
  .clear-btn:hover { background: rgba(255,255,255,0.13); color: #fff; }
  .messages {
    flex: 1; overflow-y: auto; padding: 20px 22px 10px;
    display: flex; flex-direction: column; gap: 14px; scroll-behavior: smooth;
  }
  .messages::-webkit-scrollbar { width: 4px; }
  .messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.14); border-radius: 4px; }
  .msg { display: flex; gap: 10px; align-items: flex-start; max-width: 90%; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.bot  { align-self: flex-start; }
  .avatar {
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; margin-top: 2px;
  }
  .msg.bot  .avatar { background: linear-gradient(135deg, #a78bfa, #7c3aed); }
  .msg.user .avatar { background: linear-gradient(135deg, #34d399, #059669); }
  .bubble {
    padding: 11px 15px; border-radius: 18px;
    font-size: 13.5px; line-height: 1.65; word-break: break-word;
  }
  .msg.bot  .bubble { background: rgba(255,255,255,0.09); color: #e2e8f0; border-top-left-radius: 4px; }
  .msg.user .bubble { background: linear-gradient(135deg, #7c3aed, #6d28d9); color: #fff; border-top-right-radius: 4px; }
  .bubble strong { color: #c4b5fd; font-weight: 600; }
  .msg.user .bubble strong { color: #ddd6fe; }
  .bubble em { color: rgba(255,255,255,0.48); font-style: italic; }
  .bubble hr { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 10px 0; }
  .suggestions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .sug-chip {
    padding: 5px 12px; background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.22); border-radius: 20px;
    color: #c4b5fd; font-size: 11.5px; cursor: pointer; transition: background 0.2s;
  }
  .sug-chip:hover { background: rgba(167,139,250,0.22); }
  .ts { font-size: 10px; color: rgba(255,255,255,0.24); margin-top: 3px; }
  .msg.user .ts { text-align: right; }
  .typing-indicator { display: none; align-self: flex-start; align-items: center; gap: 10px; }
  .typing-indicator.visible { display: flex; }
  .dots {
    display: flex; gap: 4px; padding: 11px 15px;
    background: rgba(255,255,255,0.09); border-radius: 18px; border-top-left-radius: 4px;
  }
  .dot {
    width: 6px; height: 6px; background: rgba(167,139,250,0.7);
    border-radius: 50%; animation: bounce 1.2s infinite;
  }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)} }
  .input-area {
    padding: 14px 18px; background: rgba(255,255,255,0.04);
    border-top: 1px solid rgba(255,255,255,0.08);
    display: flex; gap: 10px; align-items: flex-end;
  }
  .input-area textarea {
    flex: 1; background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12); border-radius: 14px;
    padding: 11px 15px; color: #fff; font-size: 13.5px;
    resize: none; outline: none; line-height: 1.5; max-height: 130px;
    font-family: inherit; transition: border-color 0.2s;
  }
  .input-area textarea::placeholder { color: rgba(255,255,255,0.28); }
  .input-area textarea:focus { border-color: rgba(167,139,250,0.5); }
  .send-btn {
    width: 42px; height: 42px; background: linear-gradient(135deg, #a78bfa, #7c3aed);
    border: none; border-radius: 12px; color: #fff; font-size: 17px;
    cursor: pointer; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    transition: transform 0.15s, opacity 0.15s;
  }
  .send-btn:hover { transform: scale(1.06); }
  .send-btn:active { transform: scale(0.94); }
  .send-btn:disabled { opacity: 0.38; cursor: not-allowed; transform: none; }
  .chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 6px; }
  .chip {
    padding: 6px 13px; background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.22); border-radius: 20px;
    color: #c4b5fd; font-size: 11.5px; cursor: pointer; transition: background 0.2s;
  }
  .chip:hover { background: rgba(167,139,250,0.22); }
</style>
</head>
<body>
<div class="chat-wrapper">
  <div class="header">
    <div class="header-left">
      <div class="header-icon">🤖</div>
      <div class="header-text">
        <h1>TechSupport Hub</h1>
        <p><span class="status-dot"></span>Online · Advanced Support Bot</p>
      </div>
    </div>
    <button class="clear-btn" onclick="clearChat()">🗑 Clear chat</button>
  </div>

  <div class="messages" id="messages">
    <div class="msg bot">
      <div class="avatar">🤖</div>
      <div>
        <div class="bubble">Hey there! 👋 Welcome to <strong>TechSupport Hub</strong>.

I'm here to help with account issues, technical problems, billing, features, and more.

What can I help you with today?</div>
        <div class="chips" id="chips">
          <span class="chip" onclick="quickSend('I cant login to my account')">🔐 Can't login</span>
          <span class="chip" onclick="quickSend('my app keeps crashing')">📱 App crashing</span>
          <span class="chip" onclick="quickSend('i want to cancel and get a refund')">💳 Refund / Cancel</span>
          <span class="chip" onclick="quickSend('how do i export my data')">📤 Export data</span>
          <span class="chip" onclick="quickSend('how much does the pro plan cost')">💰 Pricing</span>
          <span class="chip" onclick="quickSend('i want to talk to a human')">🎧 Talk to support</span>
        </div>
        <div class="ts">Just now</div>
      </div>
    </div>
    <div class="typing-indicator" id="typing">
      <div class="avatar" style="background:linear-gradient(135deg,#a78bfa,#7c3aed)">🤖</div>
      <div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
    </div>
  </div>

  <div class="input-area">
    <textarea id="userInput" placeholder="Describe your issue…" rows="1"
      onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
    <button class="send-btn" id="sendBtn" onclick="sendMessage()" title="Send (Enter)">&#10148;</button>
  </div>
</div>

<script>
  const messagesEl = document.getElementById('messages');
  const inputEl    = document.getElementById('userInput');
  const sendBtn    = document.getElementById('sendBtn');
  const typingEl   = document.getElementById('typing');
  const SESSION_ID = 'sess_' + Math.random().toString(36).substr(2, 9);

  function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }
  function now() { return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}); }

  function formatText(text) {
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g,     '<em>$1</em>');
    text = text.replace(/\n---\n/g,       '\n<hr>\n');
    return text;
  }

  function addMessage(text, isUser, suggestions) {
    const wrapper = document.createElement('div');
    wrapper.className = 'msg ' + (isUser ? 'user' : 'bot');
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? '🧑' : '🤖';
    const right = document.createElement('div');
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = formatText(text);
    right.appendChild(bubble);
    if (!isUser && suggestions && suggestions.length) {
      const sugDiv = document.createElement('div');
      sugDiv.className = 'suggestions';
      suggestions.forEach(s => {
        const chip = document.createElement('span');
        chip.className = 'sug-chip';
        chip.textContent = s.label;
        chip.onclick = () => quickSend(s.query);
        sugDiv.appendChild(chip);
      });
      right.appendChild(sugDiv);
    }
    const ts = document.createElement('div');
    ts.className = 'ts';
    ts.textContent = now();
    right.appendChild(ts);
    wrapper.appendChild(avatar);
    wrapper.appendChild(right);
    messagesEl.insertBefore(wrapper, typingEl);
    scrollToBottom();
  }

  function showTyping() { typingEl.classList.add('visible'); scrollToBottom(); }
  function hideTyping() { typingEl.classList.remove('visible'); }

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;
    const chips = document.getElementById('chips');
    if (chips) chips.remove();
    inputEl.value = '';
    autoResize(inputEl);
    sendBtn.disabled = true;
    addMessage(text, true, null);
    showTyping();
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ message: text, session_id: SESSION_ID })
      });
      const data = await res.json();
      hideTyping();
      addMessage(data.response || 'Sorry, something went wrong.', false, data.suggestions || []);
    } catch(e) {
      hideTyping();
      addMessage('Oops! Something went wrong on my end. Please try again.', false, null);
    }
    sendBtn.disabled = false;
    inputEl.focus();
  }

  function quickSend(text) { inputEl.value = text; sendMessage(); }

  function clearChat() {
    fetch('/api/clear', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ session_id: SESSION_ID })
    });
    [...messagesEl.children].forEach(el => { if (el !== typingEl) el.remove(); });
    addMessage('Chat cleared! How can I help you today?', false, null);
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }

  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 130) + 'px';
  }

  inputEl.focus();
</script>
</body>
</html>"""


@flask_app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@flask_app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data       = request.json
        message    = data.get("message", "").strip()
        session_id = data.get("session_id", "default")
        if not message:
            return jsonify({"error": "Empty message"}), 400
        response, suggestions = bot.chat(message, session_id)
        return jsonify({
            "response":    response,
            "suggestions": suggestions,
            "timestamp":   datetime.now().strftime("%H:%M"),
        })
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@flask_app.route("/api/history", methods=["GET"])
def get_history():
    session_id = request.args.get("session_id", "default")
    return jsonify({"history": bot.get_history(session_id)})


@flask_app.route("/api/clear", methods=["POST"])
def clear_history():
    data       = request.json or {}
    session_id = data.get("session_id", "default")
    bot.clear_session(session_id)
    return jsonify({"message": "Cleared"})


if __name__ == "__main__":
    print("🚀  Open http://localhost:5000 in your browser")
    print("    Press Ctrl+C to stop\n")
    flask_app.run(debug=True, host="0.0.0.0", port=5000)