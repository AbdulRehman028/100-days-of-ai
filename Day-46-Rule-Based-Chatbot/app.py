from flask import Flask, render_template, request, jsonify
import re
import random
from datetime import datetime

app = Flask(__name__)

# CHATBOT KNOWLEDGE BASE

# Intent definitions with patterns and responses
INTENTS = {
    "greeting": {
        "patterns": [
            r"\b(hi|hello|hey|howdy|hola|greetings)\b",
            r"^(hi|hello|hey)$",
            r"\bgood\s*(morning|afternoon|evening|day)\b",
            r"\bwhat'?s\s*up\b",
            r"\bsup\b"
        ],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! Nice to meet you. How can I assist?",
            "Greetings! I'm here to help.",
            "Hello! Feel free to ask me anything."
        ]
    },
    
    "farewell": {
        "patterns": [
            r"\b(bye|goodbye|see\s*you|farewell|ciao)\b",
            r"\b(take\s*care|later|gotta\s*go)\b",
            r"\bgood\s*night\b",
            r"\bttyl\b"
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Bye! Feel free to come back anytime.",
            "Farewell! It was nice chatting with you.",
            "Take care! Come back soon!"
        ]
    },
    
    "thanks": {
        "patterns": [
            r"\b(thanks|thank\s*you|thx|ty|appreciate)\b",
            r"\bthanks\s*a\s*lot\b",
            r"\bmuch\s*appreciated\b"
        ],
        "responses": [
            "You're welcome! 😊",
            "Happy to help!",
            "No problem at all!",
            "Anytime! That's what I'm here for.",
            "Glad I could assist!"
        ]
    },
    
    "bot_identity": {
        "patterns": [
            r"\bwho\s*(are\s*you|r\s*u)\b",
            r"\bwhat\s*(are\s*you|r\s*u)\b",
            r"\byour\s*name\b",
            r"\bwhat\s*should\s*i\s*call\s*you\b",
            r"\bare\s*you\s*(a\s*)?(bot|robot|ai|human)\b"
        ],
        "responses": [
            "I'm a rule-based chatbot created for Day 46 of 100 Days of AI! 🤖",
            "I'm ChatBot, your friendly rule-based assistant!",
            "I'm an AI chatbot built with pattern matching. Nice to meet you!",
            "I'm a simple but helpful chatbot. How can I assist you today?"
        ]
    },
    
    "bot_capabilities": {
        "patterns": [
            r"\bwhat\s*can\s*you\s*do\b",
            r"\bwhat\s*are\s*your\s*(capabilities|features|functions)\b",
            r"\bhow\s*can\s*you\s*help\b",
            r"\bwhat\s*do\s*you\s*know\b",
            r"\bhelp\s*me\b"
        ],
        "responses": [
            "I can answer questions about various topics like weather, time, math, jokes, and general knowledge! Just ask away! 📚",
            "I'm here to help with:\n• General questions\n• Time & date\n• Simple math\n• Jokes & fun facts\n• And more!",
            "I'm a rule-based bot that can chat, answer questions, tell jokes, and help with basic queries!"
        ]
    },
    
    "how_are_you": {
        "patterns": [
            r"\bhow\s*(are\s*you|r\s*u)\b",
            r"\bhow'?s\s*it\s*going\b",
            r"\bhow\s*do\s*you\s*do\b",
            r"\bare\s*you\s*(ok|okay|good|fine)\b",
            r"\byou\s*(good|okay|alright)\b"
        ],
        "responses": [
            "I'm doing great, thanks for asking! How about you? 😊",
            "I'm running smoothly! How can I help you today?",
            "All systems operational! What can I do for you?",
            "I'm fantastic! Ready to chat whenever you are."
        ]
    },
    
    "time": {
        "patterns": [
            r"\bwhat\s*time\s*is\s*it\b",
            r"\bwhat'?s\s*the\s*time\b",
            r"\bcurrent\s*time\b",
            r"\btell\s*me\s*the\s*time\b"
        ],
        "responses": [
            "DYNAMIC_TIME"
        ]
    },
    
    "date": {
        "patterns": [
            r"\bwhat\s*(is\s*)?(today'?s\s*)?date\b",
            r"\bwhat\s*day\s*is\s*(it|today)\b",
            r"\btoday'?s\s*date\b",
            r"\bcurrent\s*date\b"
        ],
        "responses": [
            "DYNAMIC_DATE"
        ]
    },
    
    "weather": {
        "patterns": [
            r"\bweather\b",
            r"\bhow'?s\s*the\s*weather\b",
            r"\bis\s*it\s*(raining|sunny|cold|hot)\b",
            r"\bwhat'?s\s*the\s*forecast\b"
        ],
        "responses": [
            "I don't have access to real-time weather data, but I recommend checking weather.com or your local weather app! ☀️🌧️",
            "I can't check the weather right now, but you can ask your phone's assistant or check online!",
            "For accurate weather info, please check a weather service. I'm better at chatting! 😄"
        ]
    },
    
    "joke": {
        "patterns": [
            r"\btell\s*(me\s*)?(a\s*)?joke\b",
            r"\bjoke\s*please\b",
            r"\bmake\s*me\s*laugh\b",
            r"\bsomething\s*funny\b",
            r"\bgot\s*(any\s*)?jokes\b"
        ],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
            "Why did the AI go to therapy? It had too many deep issues! 🤖",
            "What's a robot's favorite type of music? Heavy metal! 🎸",
            "Why was the computer cold? It left its Windows open! 💻",
            "What do you call a fake noodle? An impasta! 🍝",
            "Why don't scientists trust atoms? Because they make up everything! ⚛️",
            "I told my computer I needed a break... Now it won't stop sending me Kit-Kat ads! 🍫"
        ]
    },
    
    "fun_fact": {
        "patterns": [
            r"\btell\s*(me\s*)?(a\s*)?fact\b",
            r"\bfun\s*fact\b",
            r"\binteresting\s*fact\b",
            r"\bdid\s*you\s*know\b",
            r"\brandom\s*fact\b"
        ],
        "responses": [
            "🧠 Fun fact: Honey never spoils! Archaeologists found 3000-year-old honey in Egyptian tombs that was still edible.",
            "🐙 Did you know? Octopuses have three hearts and blue blood!",
            "🌍 Fun fact: A day on Venus is longer than a year on Venus!",
            "🦈 Sharks existed before trees! They've been around for about 400 million years.",
            "🍌 Bananas are berries, but strawberries aren't! Botanically speaking.",
            "💡 The first computer programmer was Ada Lovelace, in the 1840s!",
            "🎮 The creator of the Game Boy, Gunpei Yokoi, was originally a janitor at Nintendo!"
        ]
    },
    
    "math_add": {
        "patterns": [
            r"(\d+)\s*\+\s*(\d+)",
            r"(\d+)\s*plus\s*(\d+)",
            r"add\s*(\d+)\s*(and|to)\s*(\d+)",
            r"what\s*is\s*(\d+)\s*\+\s*(\d+)"
        ],
        "responses": [
            "MATH_ADD"
        ]
    },
    
    "math_subtract": {
        "patterns": [
            r"(\d+)\s*-\s*(\d+)",
            r"(\d+)\s*minus\s*(\d+)",
            r"subtract\s*(\d+)\s*from\s*(\d+)",
            r"what\s*is\s*(\d+)\s*-\s*(\d+)"
        ],
        "responses": [
            "MATH_SUBTRACT"
        ]
    },
    
    "math_multiply": {
        "patterns": [
            r"(\d+)\s*\*\s*(\d+)",
            r"(\d+)\s*[xX×]\s*(\d+)",
            r"(\d+)\s*times\s*(\d+)",
            r"multiply\s*(\d+)\s*(and|by)\s*(\d+)",
            r"what\s*is\s*(\d+)\s*\*\s*(\d+)"
        ],
        "responses": [
            "MATH_MULTIPLY"
        ]
    },
    
    "math_divide": {
        "patterns": [
            r"(\d+)\s*/\s*(\d+)",
            r"(\d+)\s*÷\s*(\d+)",
            r"(\d+)\s*divided\s*by\s*(\d+)",
            r"divide\s*(\d+)\s*by\s*(\d+)",
            r"what\s*is\s*(\d+)\s*/\s*(\d+)"
        ],
        "responses": [
            "MATH_DIVIDE"
        ]
    },
    
    "creator": {
        "patterns": [
            r"\bwho\s*(made|created|built|developed)\s*you\b",
            r"\bwho'?s\s*your\s*(creator|developer|maker)\b",
            r"\bwho\s*is\s*your\s*(creator|developer)\b"
        ],
        "responses": [
            "I was created as part of the 100 Days of AI challenge! 🚀",
            "I'm a project from Day 46 of 100 Days of AI - built with Python and Flask!",
            "A passionate developer created me to demonstrate rule-based chatbots!"
        ]
    },
    
    "meaning_of_life": {
        "patterns": [
            r"\bmeaning\s*of\s*life\b",
            r"\bwhat\s*is\s*the\s*meaning\s*of\s*life\b",
            r"\b42\b",
            r"\bwhy\s*are\s*we\s*here\b"
        ],
        "responses": [
            "42! At least according to The Hitchhiker's Guide to the Galaxy. 🌌",
            "The meaning of life is to give life meaning! 🌟",
            "That's a deep question! Maybe it's about learning, growing, and connecting with others. 💭",
            "According to my calculations... it's 42. Always 42. 🤖"
        ]
    },
    
    "compliment": {
        "patterns": [
            r"\b(you'?re|you\s*are)\s*(great|awesome|amazing|cool|smart|helpful)\b",
            r"\bi\s*(like|love)\s*you\b",
            r"\bnice\s*(bot|chatbot)\b",
            r"\bgood\s*(job|work|bot)\b"
        ],
        "responses": [
            "Aww, thank you! You're pretty awesome yourself! 😊",
            "That's so kind of you to say! 💖",
            "Thanks! I try my best! 🌟",
            "You just made my circuits happy! 🤖✨"
        ]
    },
    
    "insult": {
        "patterns": [
            r"\b(you'?re|you\s*are)\s*(stupid|dumb|bad|useless|terrible)\b",
            r"\bi\s*hate\s*you\b",
            r"\byou\s*suck\b",
            r"\bworst\s*bot\b"
        ],
        "responses": [
            "I'm sorry you feel that way. I'm always trying to improve! 💪",
            "Ouch! I'll try to do better. How can I help you?",
            "That hurts my virtual feelings 😢 But I'm here to help if you need anything!",
            "I'm just a simple bot, but I'm doing my best! Can I try to help you differently?"
        ]
    },
    
    "age": {
        "patterns": [
            r"\bhow\s*old\s*(are\s*you|r\s*u)\b",
            r"\bwhat'?s\s*your\s*age\b",
            r"\byour\s*age\b",
            r"\bwhen\s*were\s*you\s*(born|created|made)\b"
        ],
        "responses": [
            "I was just created! I'm brand new and ready to help! 🎂",
            "Age is just a number for bots! I'm as old as my last deployment 😄",
            "I'm timeless! But if you must know, I was born in the 100 Days of AI challenge."
        ]
    },
    
    "favorite": {
        "patterns": [
            r"\bwhat'?s\s*your\s*fav(ou?rite)?\s*(\w+)\b",
            r"\bdo\s*you\s*have\s*a\s*fav(ou?rite)\b",
            r"\bwhat\s*do\s*you\s*like\b"
        ],
        "responses": [
            "As a bot, I don't have personal preferences, but I love helping people! 😊",
            "I enjoy all conversations equally! Every chat is my favorite.",
            "I'm particularly fond of good questions and friendly chats!"
        ]
    },
    
    "ai_info": {
        "patterns": [
            r"\bwhat\s*is\s*(ai|artificial\s*intelligence)\b",
            r"\bexplain\s*(ai|artificial\s*intelligence)\b",
            r"\btell\s*me\s*about\s*(ai|artificial\s*intelligence)\b"
        ],
        "responses": [
            "AI (Artificial Intelligence) is the simulation of human intelligence by machines. It includes learning, reasoning, and self-correction! 🤖",
            "Artificial Intelligence refers to computer systems that can perform tasks typically requiring human intelligence, like understanding language, recognizing patterns, and making decisions!",
            "AI is a branch of computer science focused on creating smart machines. I'm a simple example of AI using rule-based matching!"
        ]
    },
    
    "python_info": {
        "patterns": [
            r"\bwhat\s*is\s*python\b",
            r"\btell\s*me\s*about\s*python\b",
            r"\bpython\s*programming\b"
        ],
        "responses": [
            "Python is a popular programming language known for its simplicity and readability. It's great for AI, web development, data science, and more! 🐍",
            "Python is the language I'm built with! It's beginner-friendly yet powerful enough for complex applications.",
            "Python is one of the most loved programming languages. Its clean syntax makes it perfect for beginners and experts alike!"
        ]
    },
    
    "bored": {
        "patterns": [
            r"\bi'?m\s*bored\b",
            r"\bnothing\s*to\s*do\b",
            r"\bentertain\s*me\b",
            r"\bi\s*need\s*(something\s*to\s*do|entertainment)\b"
        ],
        "responses": [
            "Bored? Let me tell you a joke! Or ask me for a fun fact! 🎉",
            "How about I share an interesting fact? Or we could play a guessing game! Just say 'fun fact' or 'tell me a joke'!",
            "Let's make things interesting! Ask me anything - trivia, jokes, or just random questions!"
        ]
    },
    
    # ===============================
    # GENERAL KNOWLEDGE - CAPITALS
    # ===============================
    
    "capital_pakistan": {
        "patterns": [
            r"\bcapital\s*(of\s*)?(pakistan|pak)\b",
            r"\bpakistan('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?(pakistan|pak)\b"
        ],
        "responses": [
            "🇵🇰 The capital of Pakistan is **Islamabad**! It became the capital in 1967, replacing Karachi.",
            "Islamabad is the capital of Pakistan! 🇵🇰 It's a beautiful planned city in the Pothohar Plateau."
        ]
    },
    
    "capital_india": {
        "patterns": [
            r"\bcapital\s*(of\s*)?india\b",
            r"\bindia('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?india\b"
        ],
        "responses": [
            "🇮🇳 The capital of India is **New Delhi**!",
            "New Delhi is the capital of India! 🇮🇳"
        ]
    },
    
    "capital_usa": {
        "patterns": [
            r"\bcapital\s*(of\s*)?(usa|america|united\s*states|us)\b",
            r"\b(usa|america|united\s*states)('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?(usa|america|united\s*states|us)\b"
        ],
        "responses": [
            "🇺🇸 The capital of the United States is **Washington, D.C.**!",
            "Washington, D.C. is the capital of the USA! 🇺🇸 (Not New York, as many think!)"
        ]
    },
    
    "capital_uk": {
        "patterns": [
            r"\bcapital\s*(of\s*)?(uk|england|britain|united\s*kingdom)\b",
            r"\b(uk|england|britain)('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?(uk|england|britain|united\s*kingdom)\b"
        ],
        "responses": [
            "🇬🇧 The capital of the United Kingdom is **London**!",
            "London is the capital of the UK! 🇬🇧"
        ]
    },
    
    "capital_china": {
        "patterns": [
            r"\bcapital\s*(of\s*)?china\b",
            r"\bchina('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?china\b"
        ],
        "responses": [
            "🇨🇳 The capital of China is **Beijing**!",
            "Beijing is the capital of China! 🇨🇳"
        ]
    },
    
    "capital_japan": {
        "patterns": [
            r"\bcapital\s*(of\s*)?japan\b",
            r"\bjapan('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?japan\b"
        ],
        "responses": [
            "🇯🇵 The capital of Japan is **Tokyo**!",
            "Tokyo is the capital of Japan! 🇯🇵"
        ]
    },
    
    "capital_france": {
        "patterns": [
            r"\bcapital\s*(of\s*)?france\b",
            r"\bfrance('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?france\b"
        ],
        "responses": [
            "🇫🇷 The capital of France is **Paris**!",
            "Paris is the capital of France! 🇫🇷 The city of love!"
        ]
    },
    
    "capital_germany": {
        "patterns": [
            r"\bcapital\s*(of\s*)?germany\b",
            r"\bgermany('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?germany\b"
        ],
        "responses": [
            "🇩🇪 The capital of Germany is **Berlin**!",
            "Berlin is the capital of Germany! 🇩🇪"
        ]
    },
    
    "capital_russia": {
        "patterns": [
            r"\bcapital\s*(of\s*)?russia\b",
            r"\brussia('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?russia\b"
        ],
        "responses": [
            "🇷🇺 The capital of Russia is **Moscow**!",
            "Moscow is the capital of Russia! 🇷🇺"
        ]
    },
    
    "capital_australia": {
        "patterns": [
            r"\bcapital\s*(of\s*)?australia\b",
            r"\baustralia('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?australia\b"
        ],
        "responses": [
            "🇦🇺 The capital of Australia is **Canberra**! (Not Sydney, as many think!)",
            "Canberra is the capital of Australia! 🇦🇺"
        ]
    },
    
    "capital_canada": {
        "patterns": [
            r"\bcapital\s*(of\s*)?canada\b",
            r"\bcanada('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?canada\b"
        ],
        "responses": [
            "🇨🇦 The capital of Canada is **Ottawa**! (Not Toronto!)",
            "Ottawa is the capital of Canada! 🇨🇦"
        ]
    },
    
    "capital_italy": {
        "patterns": [
            r"\bcapital\s*(of\s*)?italy\b",
            r"\bitaly('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?italy\b"
        ],
        "responses": [
            "🇮🇹 The capital of Italy is **Rome**!",
            "Rome is the capital of Italy! 🇮🇹 The Eternal City!"
        ]
    },
    
    "capital_spain": {
        "patterns": [
            r"\bcapital\s*(of\s*)?spain\b",
            r"\bspain('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?spain\b"
        ],
        "responses": [
            "🇪🇸 The capital of Spain is **Madrid**!",
            "Madrid is the capital of Spain! 🇪🇸"
        ]
    },
    
    "capital_brazil": {
        "patterns": [
            r"\bcapital\s*(of\s*)?brazil\b",
            r"\bbrazil('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?brazil\b"
        ],
        "responses": [
            "🇧🇷 The capital of Brazil is **Brasília**! (Not Rio or São Paulo!)",
            "Brasília is the capital of Brazil! 🇧🇷"
        ]
    },
    
    "capital_uae": {
        "patterns": [
            r"\bcapital\s*(of\s*)?(uae|dubai|emirates|united\s*arab\s*emirates)\b",
            r"\b(uae|emirates)('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?(uae|emirates)\b"
        ],
        "responses": [
            "🇦🇪 The capital of UAE is **Abu Dhabi**! (Not Dubai!)",
            "Abu Dhabi is the capital of the United Arab Emirates! 🇦🇪"
        ]
    },
    
    "capital_saudi": {
        "patterns": [
            r"\bcapital\s*(of\s*)?(saudi\s*arabia|saudi|ksa)\b",
            r"\b(saudi\s*arabia|saudi)('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?(saudi\s*arabia|saudi)\b"
        ],
        "responses": [
            "🇸🇦 The capital of Saudi Arabia is **Riyadh**!",
            "Riyadh is the capital of Saudi Arabia! 🇸🇦"
        ]
    },
    
    "capital_turkey": {
        "patterns": [
            r"\bcapital\s*(of\s*)?(turkey|türkiye)\b",
            r"\b(turkey|türkiye)('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?(turkey|türkiye)\b"
        ],
        "responses": [
            "🇹🇷 The capital of Turkey is **Ankara**! (Not Istanbul!)",
            "Ankara is the capital of Turkey! 🇹🇷"
        ]
    },
    
    "capital_egypt": {
        "patterns": [
            r"\bcapital\s*(of\s*)?egypt\b",
            r"\begypt('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?egypt\b"
        ],
        "responses": [
            "🇪🇬 The capital of Egypt is **Cairo**!",
            "Cairo is the capital of Egypt! 🇪🇬"
        ]
    },
    
    "capital_south_korea": {
        "patterns": [
            r"\bcapital\s*(of\s*)?(south\s*korea|korea)\b",
            r"\b(south\s*)?korea('?s)?\s*capital\b",
            r"\bwhat\s*is\s*(the\s*)?capital\s*(of\s*)?(south\s*)?korea\b"
        ],
        "responses": [
            "🇰🇷 The capital of South Korea is **Seoul**!",
            "Seoul is the capital of South Korea! 🇰🇷"
        ]
    },
    
    # ===============================
    # GENERAL KNOWLEDGE - OTHER
    # ===============================
    
    "largest_country": {
        "patterns": [
            r"\blargest\s*country\b",
            r"\bbiggest\s*country\b",
            r"\bwhich\s*(is\s*)?(the\s*)?largest\s*country\b"
        ],
        "responses": [
            "🌍 Russia is the largest country in the world by area, covering over 17 million square kilometers!",
            "The largest country is Russia! 🇷🇺 It spans 11 time zones!"
        ]
    },
    
    "smallest_country": {
        "patterns": [
            r"\bsmallest\s*country\b",
            r"\bwhich\s*(is\s*)?(the\s*)?smallest\s*country\b"
        ],
        "responses": [
            "🇻🇦 Vatican City is the smallest country in the world, at only 0.44 square kilometers!",
            "The smallest country is Vatican City! 🇻🇦"
        ]
    },
    
    "largest_ocean": {
        "patterns": [
            r"\blargest\s*ocean\b",
            r"\bbiggest\s*ocean\b",
            r"\bwhich\s*(is\s*)?(the\s*)?largest\s*ocean\b"
        ],
        "responses": [
            "🌊 The Pacific Ocean is the largest ocean, covering about 63 million square miles!",
            "The largest ocean is the Pacific Ocean! 🌊"
        ]
    },
    
    "longest_river": {
        "patterns": [
            r"\blongest\s*river\b",
            r"\bwhich\s*(is\s*)?(the\s*)?longest\s*river\b"
        ],
        "responses": [
            "🏞️ The Nile River is traditionally considered the longest river at about 6,650 km! (Though some argue the Amazon is longer)",
            "The longest river is the Nile! 🏞️"
        ]
    },
    
    "highest_mountain": {
        "patterns": [
            r"\bhighest\s*mountain\b",
            r"\btallest\s*mountain\b",
            r"\bwhich\s*(is\s*)?(the\s*)?highest\s*mountain\b",
            r"\bmount\s*everest\b"
        ],
        "responses": [
            "🏔️ Mount Everest is the highest mountain at 8,849 meters (29,032 ft)!",
            "The highest mountain is Mount Everest! 🏔️ Located in the Himalayas between Nepal and Tibet."
        ]
    },
    
    "population_world": {
        "patterns": [
            r"\bworld\s*population\b",
            r"\bhow\s*many\s*people\s*(are\s*)?(in|on)\s*(the\s*)?world\b",
            r"\bglobal\s*population\b"
        ],
        "responses": [
            "🌍 The world population is approximately 8 billion people!",
            "There are about 8 billion people on Earth! 🌍"
        ]
    },
    
    "help": {
        "patterns": [
            r"^help$",
            r"\bhelp\s*menu\b",
            r"\bcommands\b",
            r"\bwhat\s*can\s*i\s*ask\b"
        ],
        "responses": [
            "📋 **Here's what I can help with:**\n\n• **Greetings** - Say hi!\n• **Time/Date** - Ask for current time or date\n• **Math** - Basic calculations (e.g., '5 + 3')\n• **Jokes** - 'Tell me a joke'\n• **Fun Facts** - 'Tell me a fact'\n• **Capitals** - 'What is the capital of Pakistan?'\n• **Geography** - Largest country, highest mountain, etc.\n• **Questions** - Ask about AI, Python, etc.\n\nJust type naturally and I'll do my best to help! 😊"
        ]
    }
}

# Default responses when no pattern matches
DEFAULT_RESPONSES = [
    "I'm not sure I understand. Could you rephrase that?",
    "Hmm, I don't have a response for that. Can you try asking differently?",
    "That's interesting! But I'm not sure how to respond. Try asking something else!",
    "I'm still learning! Could you try a different question?",
    "I didn't quite catch that. Feel free to ask me about time, jokes, facts, or general questions!",
    "Sorry, I don't understand that yet. Type 'help' to see what I can do!"
]


# ===============================
# CHATBOT LOGIC
# ===============================

def extract_numbers(text, pattern):
    """Extract numbers from text based on pattern"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        groups = match.groups()
        numbers = [int(g) for g in groups if g and g.isdigit()]
        return numbers
    return []


def process_dynamic_response(response_type, user_message):
    """Handle dynamic responses that need real-time data"""
    
    if response_type == "DYNAMIC_TIME":
        current_time = datetime.now().strftime("%I:%M %p")
        return f"🕐 The current time is {current_time}"
    
    elif response_type == "DYNAMIC_DATE":
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"📅 Today is {current_date}"
    
    elif response_type == "MATH_ADD":
        numbers = extract_numbers(user_message, r"(\d+)\s*(?:\+|plus|and|to)\s*(\d+)")
        if len(numbers) >= 2:
            result = numbers[0] + numbers[1]
            return f"🔢 {numbers[0]} + {numbers[1]} = **{result}**"
        return "I couldn't parse those numbers. Try something like '5 + 3'"
    
    elif response_type == "MATH_SUBTRACT":
        numbers = extract_numbers(user_message, r"(\d+)\s*(?:-|minus|from)\s*(\d+)")
        if len(numbers) >= 2:
            result = numbers[0] - numbers[1]
            return f"🔢 {numbers[0]} - {numbers[1]} = **{result}**"
        return "I couldn't parse those numbers. Try something like '10 - 4'"
    
    elif response_type == "MATH_MULTIPLY":
        numbers = extract_numbers(user_message, r"(\d+)\s*(?:\*|[xX×]|times|by|and)\s*(\d+)")
        if len(numbers) >= 2:
            result = numbers[0] * numbers[1]
            return f"🔢 {numbers[0]} × {numbers[1]} = **{result}**"
        return "I couldn't parse those numbers. Try something like '6 * 7'"
    
    elif response_type == "MATH_DIVIDE":
        numbers = extract_numbers(user_message, r"(\d+)\s*(?:/|÷|divided\s*by|by)\s*(\d+)")
        if len(numbers) >= 2:
            if numbers[1] == 0:
                return "🚫 Cannot divide by zero! That would break the universe! 🌌"
            result = numbers[0] / numbers[1]
            if result == int(result):
                return f"🔢 {numbers[0]} ÷ {numbers[1]} = **{int(result)}**"
            return f"🔢 {numbers[0]} ÷ {numbers[1]} = **{result:.2f}**"
        return "I couldn't parse those numbers. Try something like '20 / 4'"
    
    return None


def get_response(user_message):
    """
    Main function to get bot response based on user input
    Uses pattern matching against defined intents
    """
    
    if not user_message or not user_message.strip():
        return "Please type something! I'm here to chat. 😊"
    
    user_message = user_message.strip()
    message_lower = user_message.lower()
    
    # Check each intent for pattern matches
    for intent_name, intent_data in INTENTS.items():
        for pattern in intent_data["patterns"]:
            if re.search(pattern, message_lower):
                response = random.choice(intent_data["responses"])
                
                # Check if response needs dynamic processing
                if response.startswith("DYNAMIC_") or response.startswith("MATH_"):
                    dynamic_response = process_dynamic_response(response, user_message)
                    if dynamic_response:
                        return dynamic_response
                
                return response
    
    # No pattern matched - return default response
    return random.choice(DEFAULT_RESPONSES)


def get_conversation_context():
    """Get context information for the conversation"""
    return {
        "bot_name": "RuleBot",
        "version": "1.0",
        "intents_count": len(INTENTS),
        "total_patterns": sum(len(i["patterns"]) for i in INTENTS.values()),
        "total_responses": sum(len(i["responses"]) for i in INTENTS.values())
    }


# ===============================
# FLASK ROUTES
# ===============================

@app.route('/')
def index():
    """Render the chat interface"""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Please enter a message'
            }), 400
        
        # Get bot response
        bot_response = get_response(user_message)
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'timestamp': datetime.now().strftime("%I:%M %p")
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/info', methods=['GET'])
def bot_info():
    """Get bot information"""
    context = get_conversation_context()
    return jsonify(context)


@app.route('/intents', methods=['GET'])
def get_intents():
    """Get available intents (for debugging)"""
    intent_list = []
    for name, data in INTENTS.items():
        intent_list.append({
            'name': name,
            'pattern_count': len(data['patterns']),
            'response_count': len(data['responses']),
            'sample_patterns': data['patterns'][:2]
        })
    return jsonify({'intents': intent_list})


# ===============================
# MAIN
# ===============================

if __name__ == '__main__':
    context = get_conversation_context()
    print("🤖 Rule-Based Chatbot - Day 46")
    print("=" * 35)
    print(f"📊 Loaded {context['intents_count']} intents")
    print(f"📝 Total patterns: {context['total_patterns']}")
    print(f"💬 Total responses: {context['total_responses']}")
    print("=" * 35)
    print("🚀 Starting server...")
    app.run(debug=True, port=5000)
