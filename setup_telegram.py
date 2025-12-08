"""
Telegram Setup Helper
Auto-detects your chat ID and tests the connection
"""
import requests
import sys

BOT_TOKEN = "8325573196:AAEI1UTia5uCgSmsoxmO3aHD3O3fV-WWF0U"


def get_chat_id():
    """Get chat ID from recent messages"""
    print("=" * 70)
    print("  TELEGRAM SETUP HELPER")
    print("=" * 70)
    print()

    print("Step 1: Make sure you've sent a message to your bot")
    print("        Go to Telegram and search for your bot")
    print("        Send any message (like 'hello' or '/start')")
    print()
    input("Press Enter once you've sent a message to your bot...")
    print()

    try:
        print("Fetching updates from Telegram...")
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if not data.get('ok'):
            print("❌ Error: Telegram API returned an error")
            print(f"   {data}")
            return None

        updates = data.get('result', [])

        if not updates:
            print("❌ No messages found!")
            print("   Please make sure you sent a message to your bot")
            print("   Then run this script again")
            return None

        # Get all unique chat IDs
        chat_ids = set()
        for update in updates:
            if 'message' in update:
                chat_id = update['message']['chat']['id']
                chat_name = update['message']['chat'].get('first_name', 'Unknown')
                chat_ids.add((chat_id, chat_name))

        if not chat_ids:
            print("❌ No chat IDs found in messages")
            return None

        print(f"✓ Found {len(chat_ids)} chat(s):")
        print()

        for idx, (chat_id, name) in enumerate(chat_ids, 1):
            print(f"  {idx}. Chat ID: {chat_id} (Name: {name})")

        print()

        if len(chat_ids) == 1:
            selected_chat_id = list(chat_ids)[0][0]
        else:
            choice = input(f"Select chat number (1-{len(chat_ids)}): ")
            try:
                idx = int(choice) - 1
                selected_chat_id = list(chat_ids)[idx][0]
            except (ValueError, IndexError):
                print("Invalid selection")
                return None

        return str(selected_chat_id)

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_message(chat_id):
    """Send a test message"""
    print()
    print("Sending test message...")

    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": "🎉 <b>Telegram Connection Successful!</b>\n\nYour trading bot is now configured to send notifications to this chat.",
            "parse_mode": "HTML"
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        print("✓ Test message sent successfully!")
        print("  Check your Telegram to see the message")
        return True

    except Exception as e:
        print(f"❌ Failed to send test message: {e}")
        return False


def update_config(chat_id):
    """Show instructions to update config"""
    print()
    print("=" * 70)
    print("  CONFIGURATION UPDATE")
    print("=" * 70)
    print()
    print("Your Chat ID:", chat_id)
    print()
    print("Option 1 - Automatic (Recommended):")
    print("  The bot will auto-detect this chat ID when you run it")
    print("  Just start the trading bot: python trading_bot.py")
    print()
    print("Option 2 - Manual Update:")
    print("  Open config.py and update this line:")
    print(f'  TELEGRAM_CHAT_ID = "{chat_id}"')
    print()
    print("=" * 70)


def main():
    """Main function"""
    chat_id = get_chat_id()

    if not chat_id:
        print()
        print("Setup failed. Please try again.")
        return False

    # Test the connection
    if test_message(chat_id):
        update_config(chat_id)
        print()
        print("✅ Telegram is ready to use!")
        print()
        print("Next steps:")
        print("  1. Run: python test_connection.py")
        print("  2. Run: python trading_bot.py")
        print()
        return True
    else:
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
