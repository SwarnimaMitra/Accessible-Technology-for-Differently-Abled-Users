import pywhatkit as kit
import warnings

def send_whatsapp_message(phone_number, message, hour=None, minute=None):
    """
    Sends a WhatsApp message instantly if hour/minute are None,
    else schedules at given time (24h format).
    """
    try:
        if hour is None or minute is None:
            # Send instantly (wait 15 seconds for page load)
            kit.sendwhatmsg_instantly(phone_number, message, wait_time=15, tab_close=True)
        else:
            kit.sendwhatmsg(phone_number, message, hour, minute)
        return True
    except Exception as e:
        warnings.warn(f"WhatsApp error: {e}")
        raise e