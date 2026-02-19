"""
Test Script for IMPROVED TechSupport Hub Bot
Verify 85-95% accuracy with enhanced training
"""

from app import ImprovedTechSupportBot
import sys

def test_improved_bot():
    """Test the improved bot with diverse queries"""
    
    print("\n" + "="*80)
    print("IMPROVED TECHSUPPORT HUB BOT - ACCURACY TEST")
    print("="*80 + "\n")
    
    # Initialize bot
    print("Initializing improved bot...")
    bot = ImprovedTechSupportBot(confidence_threshold=0.55)
    print("\n✅ Bot initialized!\n")
    
    # Test cases with expected intent
    test_cases = [
        # Account Access
        ("i cant login", "account_access"),
        ("forgot my password", "account_access"),
        ("account is locked", "account_access"),
        ("cant sign in", "account_access"),
        ("login not working", "account_access"),
        ("reset password", "account_access"),
        ("authentication failed", "account_access"),
        ("2fa not working", "account_access"),
        
        # Technical Issues  
        ("app keeps crashing", "technical_issue"),
        ("app is very slow", "technical_issue"),
        ("getting error message", "technical_issue"),
        ("sync not working", "technical_issue"),
        ("connection failed", "technical_issue"),
        ("wont load", "technical_issue"),
        ("app freezes", "technical_issue"),
        ("server error", "technical_issue"),
        
        # Billing
        ("how much does it cost", "billing"),
        ("cancel my subscription", "billing"),
        ("i want a refund", "billing"),
        ("payment failed", "billing"),
        ("charged twice", "billing"),
        ("need invoice", "billing"),
        ("pricing information", "billing"),
        ("what are the plans", "billing"),
        
        # Feature Help
        ("how do i export data", "feature_help"),
        ("share with team", "feature_help"),
        ("create backup", "feature_help"),
        ("how to sync", "feature_help"),
        ("import file", "feature_help"),
        ("use api", "feature_help"),
        ("keyboard shortcuts", "feature_help"),
        ("mobile app", "feature_help"),
        
        # Account Management
        ("delete my account", "account_management"),
        ("change my email", "account_management"),
        ("update profile", "account_management"),
        ("change password", "account_management"),
        ("enable 2fa", "account_management"),
        ("privacy settings", "account_management"),
        
        # Data Export/Privacy
        ("export all my data", "data_export"),
        ("privacy policy", "data_export"),
        ("who can see my data", "data_export"),
        ("data security", "data_export"),
        ("gdpr request", "data_export"),
        
        # General Inquiry
        ("what features do you have", "general_inquiry"),
        ("tell me about the product", "general_inquiry"),
        ("system requirements", "general_inquiry"),
        ("mobile app available", "general_inquiry"),
        ("what platforms", "general_inquiry"),
        
        # Feedback
        ("this is amazing i love it", "feedback_positive"),
        ("terrible service disappointed", "feedback_negative"),
        ("thank you so much", "feedback_positive"),
        ("this sucks", "feedback_negative"),
        
        # Out of Scope
        ("need custom development", "out_of_scope"),
        ("business partnership", "out_of_scope"),
        ("job application", "out_of_scope"),
        
        # Greetings/Farewells
        ("hello", "greeting"),
        ("hi there", "greeting"),
        ("goodbye", "farewell"),
        ("thanks bye", "farewell"),
    ]
    
    print("="*80)
    print(f"Testing {len(test_cases)} diverse customer queries...")
    print("="*80 + "\n")
    
    correct = 0
    total = len(test_cases)
    failed_cases = []
    low_confidence = []
    
    for i, (message, expected_intent) in enumerate(test_cases, 1):
        result = bot.get_detailed_analysis(message)
        predicted = result['intent']
        confidence = result['intent_confidence']
        
        # Check if correct
        is_correct = (predicted == expected_intent)
        
        if is_correct:
            correct += 1
            status = "✅ PASS"
            color = ""
        else:
            status = "❌ FAIL"
            color = ""
            failed_cases.append((message, expected_intent, predicted, confidence))
        
        # Track low confidence
        if confidence < 0.65:
            low_confidence.append((message, predicted, confidence))
        
        # Print result
        print(f"[{i}/{total}] {status}")
        print(f"  Message: '{message}'")
        print(f"  Expected: {expected_intent} | Got: {predicted} | Confidence: {confidence:.1%}")
        
        if not is_correct:
            print(f"  ⚠️  Mismatch!")
        
        print()
    
    # Calculate accuracy
    accuracy = (correct / total) * 100
    
    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print(f"\n📊 Total Tests: {total}")
    print(f"✅ Correct: {correct}")
    print(f"❌ Incorrect: {total - correct}")
    print(f"\n🎯 ACCURACY: {accuracy:.1f}%\n")
    
    # Show performance grade
    if accuracy >= 90:
        print("🏆 GRADE: EXCELLENT (90%+)")
        print("    Bot is production-ready!")
    elif accuracy >= 80:
        print("⭐ GRADE: VERY GOOD (80-89%)")
        print("    Bot performs well, minor improvements possible")
    elif accuracy >= 70:
        print("👍 GRADE: GOOD (70-79%)")
        print("    Bot is functional, consider adding more training data")
    elif accuracy >= 60:
        print("⚠️  GRADE: FAIR (60-69%)")
        print("    Needs more training data and tuning")
    else:
        print("❌ GRADE: NEEDS WORK (<60%)")
        print("    Requires significant improvements")
    
    print("\n" + "="*80)
    
    # Show failed cases if any
    if failed_cases:
        print("\n❌ FAILED TEST CASES:")
        print("-" * 80)
        for msg, expected, predicted, conf in failed_cases:
            print(f"\nMessage: '{msg}'")
            print(f"  Expected: {expected}")
            print(f"  Got: {predicted} ({conf:.1%} confidence)")
    
    # Show low confidence cases
    if low_confidence:
        print("\n⚠️  LOW CONFIDENCE CASES (<65%):")
        print("-" * 80)
        for msg, intent, conf in low_confidence[:5]:  # Show first 5
            print(f"\nMessage: '{msg}'")
            print(f"  Intent: {intent} ({conf:.1%})")
    
    print("\n" + "="*80)
    
    # Training stats
    stats = bot.get_training_stats()
    if stats:
        print("\n📈 TRAINING STATISTICS:")
        print("-" * 80)
        print(f"Training Examples: {stats.get('training_size', 0)}")
        print(f"Number of Intents: {stats.get('num_intents', 0)}")
        print(f"Cross-Validation Accuracy: {stats.get('mean_accuracy', 0)*100:.2f}%")
        print(f"Confidence Interval: ±{stats.get('std_accuracy', 0)*2*100:.2f}%")
    
    print("\n" + "="*80)
    print("\n💡 RECOMMENDATIONS:")
    
    if accuracy >= 85:
        print("✅ Bot is ready for production use!")
        print("✅ Responses are detailed and helpful")
        print("✅ Can handle diverse customer phrasings")
    else:
        print("⚠️  Consider these improvements:")
        if accuracy < 80:
            print("   • Add more training examples (especially for failed cases)")
            print("   • Lower confidence threshold to 0.50")
        if len(low_confidence) > 5:
            print("   • Add more variations of low-confidence queries to training data")
        if failed_cases:
            print("   • Review failed cases and add similar examples to training")
    
    print("\n" + "="*80)
    print("\n🚀 NEXT STEPS:")
    print("1. If accuracy is good (85%+), deploy to production")
    print("2. Run the web interface: python improved_web_interface.py")
    print("3. Test manually with real customer queries")
    print("4. Monitor and retrain with new examples monthly")
    print("\n" + "="*80 + "\n")
    
    return accuracy >= 70


if __name__ == "__main__":
    try:
        success = test_improved_bot()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
