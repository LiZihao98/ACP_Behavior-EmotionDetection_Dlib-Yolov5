from datetime import datetime
def driver_warning(fatigue: bool, behav: str, emotion: str) -> str:

    negtive_emo = []
    positive_emo = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    warning = []
    warning[0] = ""
    warning[1] = False
    
    # Scenario 1: Fatigue + risky behavior + negative emotion
    if fatigue and behav != "no bad behaviours" and emotion in negtive_emo:
        warning[0] = f"[{current_time}][High Alert] You're tired, doing something risky, and in a bad moode. Please pull over and rest ASAP!"
        warning[1] = True

        return warning
    
    # Scenario 2: Fatigue + risky behavior + positive (or stable) emotion
    if fatigue and behav != "no bad behaviours" and emotion in positive_emo:
        warning[0] = f"[{current_time}][Heads Up] You're tired and doing something risky. Stay focused and stop the risky action."
        warning[1] = True

        return warning
    
    # Scenario 3: Not fatigued + risky behavior + positive emotion
    if not fatigue and behav != "no bad behaviours" and emotion in positive_emo:
        warning[0] = f"[{current_time}][Watch Out]  You're doing something risky. Don't get too excited and lose focus."
        warning[1] = False

        return warning
    
    # Scenario 4: Fatigue + negative emotion + no dangerous behavior
    if fatigue and behav == "no bad behaviours" and emotion in negtive_emo:
        warning[0] = f"[{current_time}] [Take a Break] You're tired and in a bad mood, though not doing anything risky. Consider resting or relaxing."
        warning[1] = True 

        return warning
    
    if fatigue:
        warning[0] = f"[{current_time}] You are now tired, please pullover and choose a place to rest."
        warning[1] = True

        return warning
    
    if behav != "no bad behaviours":
        warning[0] = f"[{current_time}] You now "+behav+" Please focus on driving!"
        warning[1] = False
        
        return warning

    if emotion in negtive_emo:
        warning[0] = f"[{current_time}] You are now "+emotion+" Please stay in a normal mindset and focus on driving."
        warning[1] = False

        return warning

    
    # Default scenario if none of the above conditions match
    return warning