import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    """Display a beautiful banner"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           🛡️  SPAM CLASSIFIER AI DETECTOR 🛡️              ║
║                                                           ║
║         Protect yourself from unwanted messages!         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)

def print_section(title):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}\n")

def load_and_train_model():
    """Load dataset and train the model"""
    print_section("📚 TRAINING THE AI MODEL")
    
    try:
        # Try reading with default settings first
        data = pd.read_csv("spam.csv", encoding='utf-8')
    except pd.errors.ParserError:
        # If that fails, try with error handling
        print(f"{Colors.YELLOW}⚠️  CSV has formatting issues, using alternative parsing...{Colors.ENDC}")
        data = pd.read_csv("spam.csv", encoding='utf-8', on_bad_lines='skip')

    # Clean column names (handle potential extra columns)
    if len(data.columns) > 2:
        data = data.iloc[:, :2]
        data.columns = ["label", "message"]
    else:
        data.columns = ["label", "message"]

    print(f"{Colors.GREEN}✓ Loaded {len(data)} messages{Colors.ENDC}")
    
    # Display class distribution
    ham_count = (data['label'] == 'ham').sum()
    spam_count = (data['label'] == 'spam').sum()
    print(f"{Colors.CYAN}  📊 Ham (legitimate): {ham_count}{Colors.ENDC}")
    print(f"{Colors.CYAN}  📊 Spam (unwanted): {spam_count}{Colors.ENDC}")
    
    # Encode labels
    data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label_num"], test_size=0.2, random_state=42
    )

    # Vectorize text
    print(f"\n{Colors.YELLOW}🔄 Training AI model...{Colors.ENDC}")
    vectorizer = CountVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{Colors.GREEN}✓ Model trained successfully!{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Accuracy: {accuracy*100:.1f}%{Colors.ENDC}")
    
    return model, vectorizer

def classify_message(message, model, vectorizer):
    """Classify a single message"""
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    probability = model.predict_proba(message_vec)[0]
    
    confidence = max(probability) * 100
    
    return prediction, confidence

def display_result(message, prediction, confidence):
    """Display classification result beautifully"""
    print(f"\n{Colors.BOLD}{'─'*60}{Colors.ENDC}")
    print(f"{Colors.CYAN}📩 Your Message:{Colors.ENDC}")
    print(f"   \"{message}\"")
    print(f"{Colors.BOLD}{'─'*60}{Colors.ENDC}\n")
    
    if prediction == 1:  # Spam
        print(f"{Colors.RED}{Colors.BOLD}🚨 RESULT: SPAM DETECTED! 🚨{Colors.ENDC}")
        print(f"{Colors.RED}   This message appears to be spam/unwanted.{Colors.ENDC}")
        print(f"{Colors.YELLOW}   Confidence: {confidence:.1f}%{Colors.ENDC}")
        print(f"{Colors.RED}   ⚠️  Be cautious! Do not click links or respond.{Colors.ENDC}")
    else:  # Ham
        print(f"{Colors.GREEN}{Colors.BOLD}✅ RESULT: LEGITIMATE MESSAGE ✅{Colors.ENDC}")
        print(f"{Colors.GREEN}   This message appears to be safe.{Colors.ENDC}")
        print(f"{Colors.YELLOW}   Confidence: {confidence:.1f}%{Colors.ENDC}")
        print(f"{Colors.GREEN}   ✓ This looks like a genuine message.{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}{'─'*60}{Colors.ENDC}")

def interactive_mode(model, vectorizer):
    """Run interactive classification mode"""
    print_section("🎯 INTERACTIVE SPAM DETECTOR")
    
    print(f"{Colors.CYAN}Enter messages to check if they're spam.{Colors.ENDC}")
    print(f"{Colors.CYAN}Type 'quit' or 'exit' to stop.{Colors.ENDC}\n")
    
    while True:
        try:
            # Get user input
            print(f"{Colors.BOLD}{Colors.BLUE}Enter a message to analyze:{Colors.ENDC} ", end="")
            user_message = input().strip()
            
            # Check for exit commands
            if user_message.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.CYAN}👋 Thank you for using Spam Classifier AI!{Colors.ENDC}")
                print(f"{Colors.CYAN}Stay safe from spam! 🛡️{Colors.ENDC}\n")
                break
            
            # Skip empty messages
            if not user_message:
                print(f"{Colors.YELLOW}⚠️  Please enter a message to analyze.{Colors.ENDC}\n")
                continue
            
            # Classify the message
            prediction, confidence = classify_message(user_message, model, vectorizer)
            display_result(user_message, prediction, confidence)
            
            print()  # Add spacing
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}� Interrupted. Goodbye!{Colors.ENDC}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.ENDC}\n")

def main():
    """Main function"""
    # Clear screen for better presentation
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Display banner
    print_banner()
    
    # Train model
    model, vectorizer = load_and_train_model()
    
    # Run interactive mode
    interactive_mode(model, vectorizer)

if __name__ == "__main__":
    main()
