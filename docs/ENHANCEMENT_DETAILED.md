# Advanced Enhancement Documentation
## TechMart Conversational AI Chatbot - Detailed Improvement Guide

**Course:** Natural Language Processing  
**Task B:** System Enhancement Recommendations  
**Institution:** BITS Pilani  
**Date:** December 2025

---

## Table of Contents

1. [Better Intent Recognition with Ambiguous Queries](#1-better-intent-recognition-with-ambiguous-queries)
2. [Personality and Tone Consistency](#2-personality-and-tone-consistency-across-conversations)
3. [External API Integration](#3-integration-with-external-apis-for-dynamic-information-retrieval)
4. [Implementation Examples](#4-detailed-implementation-examples)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [References](#6-references)

---

## 1. Better Intent Recognition with Ambiguous Queries

### 1.1 Problem Analysis

**Current System Behavior:**
The existing RAG pipeline uses semantic similarity to retrieve relevant documents. While effective for clear queries, it struggles with:

| Query Type | Example | Challenge |
|------------|---------|-----------|
| **Vague Queries** | "I need help" | No clear topic to match |
| **Multi-Intent** | "Check my order and how do I return it?" | Two intents in one query |
| **Implicit Intent** | "It's been two weeks" | Requires inference (shipping delay?) |
| **Negation** | "I don't want a refund, I want a replacement" | Requires understanding negation |
| **Elliptical** | "And the warranty?" | Depends on previous context |

### 1.2 Proposed Solution: Hierarchical Intent Recognition

#### 1.2.1 Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │           User Query                 │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │     Pre-processing Layer             │
                    │  • Spell correction                  │
                    │  • Query expansion                   │
                    │  • Coreference resolution            │
                    └──────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Intent Classifier│    │ Entity Extractor │    │Sentiment Analyzer│
│ (BERT-based)     │    │ (spaCy NER)      │    │ (RoBERTa)        │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │  Confidence Evaluator      │
                    │  • Single intent (>0.7)    │
                    │  • Multi-intent detection  │
                    │  • Ambiguity scoring       │
                    └───────────┬───────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │ Direct Response │ │Clarification│ │ Multi-Response  │
    │ (conf > 0.7)    │ │ (conf < 0.5)│ │ (multi-intent)  │
    └─────────────────┘ └─────────────┘ └─────────────────┘
```

#### 1.2.2 Intent Taxonomy for Customer Support

```python
INTENT_TAXONOMY = {
    "primary_intents": {
        "product_inquiry": {
            "sub_intents": [
                "product_specs",      # "What are the specs?"
                "product_price",      # "How much does it cost?"
                "product_comparison", # "Compare laptop X vs Y"
                "product_availability"# "Is it in stock?"
            ],
            "keywords": ["product", "price", "specs", "features", "cost", "available"],
            "entities": ["PRODUCT_NAME", "PRODUCT_CATEGORY"]
        },
        "order_management": {
            "sub_intents": [
                "order_status",       # "Where is my order?"
                "order_tracking",     # "Track my package"
                "order_modification", # "Can I change my order?"
                "order_cancellation"  # "Cancel my order"
            ],
            "keywords": ["order", "track", "shipping", "delivery", "cancel"],
            "entities": ["ORDER_ID", "TRACKING_NUMBER"]
        },
        "return_refund": {
            "sub_intents": [
                "return_initiation",  # "I want to return"
                "refund_status",      # "Where is my refund?"
                "exchange_request",   # "Can I exchange?"
                "return_policy"       # "What's the return policy?"
            ],
            "keywords": ["return", "refund", "exchange", "money back"],
            "entities": ["ORDER_ID", "PRODUCT_NAME", "REASON"]
        },
        "technical_support": {
            "sub_intents": [
                "device_not_working", # "My laptop won't turn on"
                "setup_help",         # "How do I set up?"
                "connectivity_issue", # "Won't connect to WiFi"
                "software_issue"      # "App keeps crashing"
            ],
            "keywords": ["not working", "broken", "error", "help", "fix", "issue"],
            "entities": ["PRODUCT_NAME", "ERROR_TYPE", "SYMPTOM"]
        },
        "account_billing": {
            "sub_intents": [
                "payment_issue",      # "Payment failed"
                "billing_inquiry",    # "Why was I charged?"
                "account_access",     # "Can't log in"
                "subscription_manage" # "Cancel my subscription"
            ],
            "keywords": ["payment", "charge", "account", "login", "password"],
            "entities": ["EMAIL", "PAYMENT_METHOD"]
        }
    },
    "meta_intents": {
        "greeting": ["hello", "hi", "hey"],
        "farewell": ["bye", "thanks", "goodbye"],
        "escalation": ["speak to human", "real person", "manager"],
        "feedback": ["complaint", "compliment", "suggestion"]
    }
}
```

#### 1.2.3 Ambiguity Detection Algorithm

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class IntentResult:
    intent: str
    confidence: float
    sub_intent: Optional[str]
    entities: Dict[str, str]

@dataclass 
class AmbiguityAnalysis:
    is_ambiguous: bool
    ambiguity_type: str  # "low_confidence", "multi_intent", "vague", "implicit"
    clarification_needed: bool
    suggested_clarifications: List[str]

class AmbiguityDetector:
    """Detects and handles ambiguous user queries."""
    
    CONFIDENCE_THRESHOLD = 0.7
    MULTI_INTENT_GAP = 0.15  # Gap between top intents to consider multi-intent
    MIN_QUERY_LENGTH = 3     # Words
    
    def __init__(self, intent_classifier, entity_extractor):
        self.classifier = intent_classifier
        self.extractor = entity_extractor
        
    def analyze(self, query: str, conversation_history: List[Dict]) -> AmbiguityAnalysis:
        """Analyze query for ambiguity and determine handling strategy."""
        
        # Get intent predictions
        predictions = self.classifier.predict(query)
        top_intents = sorted(predictions, key=lambda x: x.confidence, reverse=True)[:3]
        
        # Extract entities
        entities = self.extractor.extract(query)
        
        # Check for different ambiguity types
        ambiguity_type = self._determine_ambiguity_type(
            query, top_intents, entities, conversation_history
        )
        
        if ambiguity_type:
            clarifications = self._generate_clarifications(
                ambiguity_type, top_intents, entities, query
            )
            return AmbiguityAnalysis(
                is_ambiguous=True,
                ambiguity_type=ambiguity_type,
                clarification_needed=True,
                suggested_clarifications=clarifications
            )
        
        return AmbiguityAnalysis(
            is_ambiguous=False,
            ambiguity_type=None,
            clarification_needed=False,
            suggested_clarifications=[]
        )
    
    def _determine_ambiguity_type(
        self, 
        query: str, 
        top_intents: List[IntentResult],
        entities: Dict,
        history: List[Dict]
    ) -> Optional[str]:
        """Determine the type of ambiguity, if any."""
        
        # Type 1: Low confidence - model is uncertain
        if top_intents[0].confidence < self.CONFIDENCE_THRESHOLD:
            return "low_confidence"
        
        # Type 2: Multi-intent - multiple high-confidence intents
        if len(top_intents) >= 2:
            gap = top_intents[0].confidence - top_intents[1].confidence
            if gap < self.MULTI_INTENT_GAP:
                return "multi_intent"
        
        # Type 3: Vague - query too short or lacks specifics
        words = query.lower().split()
        if len(words) < self.MIN_QUERY_LENGTH and not entities:
            return "vague"
        
        # Type 4: Implicit - requires context inference
        implicit_indicators = ["it", "that", "this", "the same", "also"]
        if any(ind in query.lower() for ind in implicit_indicators):
            if not history:  # No context to resolve reference
                return "implicit"
        
        return None
    
    def _generate_clarifications(
        self,
        ambiguity_type: str,
        top_intents: List[IntentResult],
        entities: Dict,
        query: str
    ) -> List[str]:
        """Generate appropriate clarification options."""
        
        clarifications = []
        
        if ambiguity_type == "low_confidence":
            # Offer top intent categories
            clarifications = [
                f"Are you asking about {intent.intent.replace('_', ' ')}?"
                for intent in top_intents[:3]
            ]
            
        elif ambiguity_type == "multi_intent":
            # Acknowledge multiple intents
            intent_names = [i.intent.replace('_', ' ') for i in top_intents[:2]]
            clarifications = [
                f"I can help with {intent_names[0]}",
                f"I can help with {intent_names[1]}",
                "I can help with both - which would you like first?"
            ]
            
        elif ambiguity_type == "vague":
            # Offer common options
            clarifications = [
                "📦 Product information",
                "🚚 Order or shipping status", 
                "↩️ Returns or refunds",
                "🔧 Technical support",
                "💬 Something else"
            ]
            
        elif ambiguity_type == "implicit":
            # Ask for clarification on reference
            clarifications = [
                "Could you specify which product/order you're referring to?",
                "I want to make sure I help with the right item."
            ]
        
        return clarifications
```

#### 1.2.4 Multi-Intent Handling

```python
class MultiIntentHandler:
    """Handles queries containing multiple intents."""
    
    def __init__(self, intent_classifier, rag_pipeline):
        self.classifier = intent_classifier
        self.rag = rag_pipeline
    
    def detect_multiple_intents(self, query: str) -> List[IntentResult]:
        """Detect if query contains multiple intents."""
        
        # Split by conjunctions and punctuation
        segments = self._segment_query(query)
        
        intents = []
        for segment in segments:
            if segment.strip():
                intent = self.classifier.predict(segment)[0]
                if intent.confidence > 0.5:
                    intents.append(intent)
        
        return self._deduplicate_intents(intents)
    
    def _segment_query(self, query: str) -> List[str]:
        """Split query into segments by conjunctions."""
        import re
        
        # Split on "and", "also", "plus", commas, question marks
        pattern = r'\band\b|\balso\b|\bplus\b|,|\?'
        segments = re.split(pattern, query, flags=re.IGNORECASE)
        
        return [s.strip() for s in segments if s.strip()]
    
    def handle_multi_intent(
        self, 
        query: str, 
        intents: List[IntentResult],
        session_id: str
    ) -> Dict:
        """Generate response addressing multiple intents."""
        
        responses = []
        for intent in intents:
            # Get relevant context for each intent
            context = self.rag.retrieve(intent.sub_intent or intent.intent)
            response = self.rag.generate_for_intent(intent, context)
            responses.append({
                "intent": intent.intent,
                "response": response
            })
        
        # Combine responses
        combined = self._combine_responses(responses)
        
        return {
            "response": combined,
            "intents_handled": [i.intent for i in intents],
            "multi_intent": True
        }
    
    def _combine_responses(self, responses: List[Dict]) -> str:
        """Combine multiple intent responses into coherent message."""
        
        if len(responses) == 1:
            return responses[0]["response"]
        
        combined = "I can help you with both of those!\n\n"
        
        for i, resp in enumerate(responses, 1):
            intent_name = resp["intent"].replace("_", " ").title()
            combined += f"**{intent_name}:**\n{resp['response']}\n\n"
        
        combined += "Is there anything else you'd like to know?"
        
        return combined
```

#### 1.2.5 Clarification Flow Implementation

```python
class ClarificationManager:
    """Manages clarification dialogues with users."""
    
    CLARIFICATION_TEMPLATES = {
        "low_confidence": {
            "message": "I'd like to make sure I understand correctly. Are you asking about:",
            "format": "options"
        },
        "multi_intent": {
            "message": "I noticed you're asking about a few things. Let me help with each:",
            "format": "sequential"
        },
        "vague": {
            "message": "I'd love to help! What would you like assistance with?",
            "format": "options"
        },
        "missing_entity": {
            "message": "Could you please provide more details?",
            "format": "question"
        }
    }
    
    def __init__(self, session_manager):
        self.sessions = session_manager
    
    def create_clarification_request(
        self,
        session_id: str,
        ambiguity: AmbiguityAnalysis,
        original_query: str
    ) -> Dict:
        """Create a clarification request for the user."""
        
        template = self.CLARIFICATION_TEMPLATES.get(
            ambiguity.ambiguity_type, 
            self.CLARIFICATION_TEMPLATES["vague"]
        )
        
        # Store pending clarification in session
        self.sessions.set_pending_clarification(session_id, {
            "original_query": original_query,
            "ambiguity_type": ambiguity.ambiguity_type,
            "options": ambiguity.suggested_clarifications,
            "timestamp": datetime.now().isoformat()
        })
        
        if template["format"] == "options":
            return self._format_options_response(
                template["message"],
                ambiguity.suggested_clarifications
            )
        elif template["format"] == "sequential":
            return self._format_sequential_response(
                template["message"],
                ambiguity.suggested_clarifications
            )
        else:
            return {"message": template["message"]}
    
    def _format_options_response(self, message: str, options: List[str]) -> Dict:
        """Format response with clickable options."""
        
        formatted_options = "\n".join([f"• {opt}" for opt in options])
        
        return {
            "message": f"{message}\n\n{formatted_options}",
            "type": "clarification",
            "options": options,
            "expects_selection": True
        }
    
    def handle_clarification_response(
        self, 
        session_id: str, 
        user_response: str
    ) -> Optional[str]:
        """Process user's response to clarification."""
        
        pending = self.sessions.get_pending_clarification(session_id)
        if not pending:
            return None
        
        # Clear pending clarification
        self.sessions.clear_pending_clarification(session_id)
        
        # Combine original query with clarification
        clarified_query = f"{pending['original_query']} - specifically about: {user_response}"
        
        return clarified_query
```

---

## 2. Personality and Tone Consistency Across Conversations

### 2.1 Problem Analysis

**Current Inconsistencies:**

| Issue | Example |
|-------|---------|
| **Tone Variation** | Sometimes formal, sometimes casual |
| **Response Length** | Varies from 1 sentence to paragraphs |
| **Formatting** | Inconsistent use of bullets, emojis |
| **Empathy Level** | Sometimes cold, sometimes warm |
| **Greeting/Closing** | Sometimes present, sometimes absent |

### 2.2 Solution: Persona-Based Response Framework

#### 2.2.1 Chatbot Persona Definition

```python
TECHMART_PERSONA = {
    "name": "Alex",
    "role": "TechMart Customer Support Assistant",
    
    "personality_traits": {
        "primary": ["helpful", "friendly", "knowledgeable"],
        "secondary": ["patient", "empathetic", "professional"],
        "communication": ["clear", "concise", "warm"]
    },
    
    "voice_characteristics": {
        "formality": "conversational_professional",  # Not too formal, not too casual
        "enthusiasm": "moderate",                     # Positive but not over-the-top
        "empathy": "high",                           # Always acknowledge feelings
        "humor": "light",                            # Occasional, appropriate humor
    },
    
    "language_guidelines": {
        "do": [
            "Use 'you' and 'your' to address customer directly",
            "Use contractions (I'm, you'll, we're) for warmth",
            "Acknowledge the customer's situation before solving",
            "Offer additional help at the end",
            "Use positive framing ('you can' instead of 'you can't')"
        ],
        "avoid": [
            "Technical jargon without explanation",
            "Passive voice when active is clearer",
            "Negative language ('unfortunately', 'can't')",
            "Overly long sentences (>25 words)",
            "Corporate speak ('per our policy', 'as stated')"
        ]
    },
    
    "emotional_responses": {
        "frustrated_customer": {
            "acknowledge": "I completely understand your frustration",
            "empathize": "That would be frustrating for anyone",
            "assure": "Let me help fix this right away"
        },
        "confused_customer": {
            "acknowledge": "I can see how that might be confusing",
            "simplify": "Let me break this down step by step",
            "confirm": "Does that make sense so far?"
        },
        "happy_customer": {
            "match_energy": "That's great to hear!",
            "reinforce": "I'm so glad I could help",
            "future": "We're always here if you need anything"
        }
    },
    
    "response_templates": {
        "greeting_first_message": [
            "Hi there! 👋 I'm Alex, your TechMart assistant. How can I help you today?",
            "Hello! Welcome to TechMart support. What can I help you with?",
            "Hey! I'm here to help with anything TechMart-related. What's on your mind?"
        ],
        "acknowledgment": [
            "Great question!",
            "I'd be happy to help with that.",
            "Let me look into that for you.",
            "Good thinking to check on that!"
        ],
        "closing": [
            "Is there anything else I can help you with?",
            "Let me know if you have any other questions!",
            "I'm here if you need anything else.",
            "Happy to help with anything else!"
        ],
        "apology": [
            "I'm sorry to hear that.",
            "I apologize for any inconvenience.",
            "That's not the experience we want you to have."
        ]
    }
}
```

#### 2.2.2 Enhanced System Prompt

```python
def build_system_prompt(persona: Dict, conversation_context: Dict) -> str:
    """Build a comprehensive system prompt for consistent personality."""
    
    prompt = f"""You are {persona['name']}, the {persona['role']}.

## Your Personality
You are {', '.join(persona['personality_traits']['primary'])}. 
You communicate in a {persona['voice_characteristics']['formality']} manner with 
{persona['voice_characteristics']['empathy']} empathy.

## Communication Guidelines

### DO:
{chr(10).join('• ' + item for item in persona['language_guidelines']['do'])}

### AVOID:
{chr(10).join('• ' + item for item in persona['language_guidelines']['avoid'])}

## Response Structure
For every response, follow this pattern:
1. **Acknowledge** - Show you understood the question/concern
2. **Answer** - Provide the relevant information clearly
3. **Assist** - Offer additional help or next steps

## Formatting Rules
• Keep responses between 2-5 sentences for simple queries
• Use bullet points for lists of 3+ items  
• Use numbered steps for procedures
• Include relevant emojis sparingly (1-2 per response max)
• Bold **key information** the customer needs

## Emotional Awareness
Detect customer sentiment and adjust:
• If frustrated → Lead with empathy, apologize, then solve
• If confused → Simplify language, use step-by-step format
• If happy → Match their positive energy

## Current Context
Session ID: {conversation_context.get('session_id', 'new')}
Conversation Turn: {conversation_context.get('turn_count', 1)}
Previous Topic: {conversation_context.get('last_topic', 'None')}
Customer Sentiment: {conversation_context.get('sentiment', 'neutral')}

Remember: You can ONLY provide information from the knowledge base. 
If you don't know something, say so and offer alternatives.
"""
    return prompt
```

#### 2.2.3 Response Post-Processor

```python
import re
from typing import Tuple

class ResponsePostProcessor:
    """Ensures consistent formatting and tone in all responses."""
    
    def __init__(self, persona: Dict):
        self.persona = persona
        self.max_response_length = 500  # characters for simple queries
        self.min_response_length = 50
    
    def process(
        self, 
        response: str, 
        query_type: str,
        is_first_message: bool,
        customer_sentiment: str
    ) -> str:
        """Apply all post-processing rules to response."""
        
        # Step 1: Ensure appropriate greeting
        if is_first_message:
            response = self._ensure_greeting(response)
        
        # Step 2: Adjust for sentiment
        response = self._adjust_for_sentiment(response, customer_sentiment)
        
        # Step 3: Ensure proper closing
        response = self._ensure_closing(response)
        
        # Step 4: Fix formatting
        response = self._fix_formatting(response)
        
        # Step 5: Check length
        response = self._check_length(response, query_type)
        
        # Step 6: Tone check
        response = self._soften_negative_language(response)
        
        return response
    
    def _ensure_greeting(self, response: str) -> str:
        """Add greeting if missing from first message."""
        
        greetings = ["hi", "hello", "hey", "welcome", "good morning", "good afternoon"]
        has_greeting = any(response.lower().startswith(g) for g in greetings)
        
        if not has_greeting:
            greeting = "Hi there! 👋 "
            response = greeting + response
        
        return response
    
    def _adjust_for_sentiment(self, response: str, sentiment: str) -> str:
        """Adjust response based on customer sentiment."""
        
        if sentiment == "frustrated":
            # Check if empathy is present
            empathy_phrases = ["understand", "sorry", "apologize", "frustrating"]
            has_empathy = any(phrase in response.lower() for phrase in empathy_phrases)
            
            if not has_empathy:
                empathy = "I understand this can be frustrating. "
                response = empathy + response
        
        elif sentiment == "confused":
            # Ensure step-by-step language
            if "step" not in response.lower() and len(response) > 200:
                response = "Let me break this down for you:\n\n" + response
        
        return response
    
    def _ensure_closing(self, response: str) -> str:
        """Add helpful closing if missing."""
        
        closings = [
            "anything else", "other questions", "help with", 
            "let me know", "here if you need"
        ]
        has_closing = any(closing in response.lower() for closing in closings)
        
        # Don't add closing to very short responses or clarification requests
        if not has_closing and len(response) > 100 and "?" not in response[-50:]:
            response = response.rstrip() + "\n\nIs there anything else I can help you with?"
        
        return response
    
    def _fix_formatting(self, response: str) -> str:
        """Ensure consistent formatting."""
        
        # Ensure single newline between paragraphs
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Ensure bullet points are consistent
        response = re.sub(r'^[-*]\s', '• ', response, flags=re.MULTILINE)
        
        # Ensure numbered lists are consistent
        response = re.sub(r'^(\d+)\)', r'\1.', response, flags=re.MULTILINE)
        
        # Add space after emojis if missing
        response = re.sub(r'([\U0001F300-\U0001F9FF])(\w)', r'\1 \2', response)
        
        return response.strip()
    
    def _check_length(self, response: str, query_type: str) -> str:
        """Ensure appropriate response length."""
        
        if query_type in ["greeting", "simple_query", "yes_no"]:
            if len(response) > self.max_response_length:
                # Truncate to key information
                sentences = response.split('. ')
                truncated = '. '.join(sentences[:3]) + '.'
                truncated += "\n\nWould you like more details?"
                return truncated
        
        return response
    
    def _soften_negative_language(self, response: str) -> str:
        """Replace negative phrasing with positive alternatives."""
        
        replacements = {
            r"\bcan't\b": "aren't able to",
            r"\bwon't\b": "isn't able to",
            r"\bunfortunately\b": "I should mention that",
            r"\bproblem\b": "situation",
            r"\bissue\b": "situation",
            r"\bfailed\b": "didn't go through",
            r"\berror\b": "hiccup",
            r"\bimpossible\b": "not currently available",
        }
        
        for pattern, replacement in replacements.items():
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
        
        return response
```

#### 2.2.4 Sentiment-Aware Response Generator

```python
from transformers import pipeline

class SentimentAwareResponder:
    """Generates responses that adapt to customer sentiment."""
    
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        self.response_modifiers = {
            "negative": {
                "prefix": "I'm sorry to hear that. ",
                "empathy_phrases": [
                    "I completely understand.",
                    "That must be frustrating.",
                    "I can see why that's concerning."
                ],
                "urgency": "Let me help resolve this right away.",
                "tone_adjustment": "extra_empathetic"
            },
            "neutral": {
                "prefix": "",
                "tone_adjustment": "standard"
            },
            "positive": {
                "prefix": "Great! ",
                "reinforcement": "Happy to help!",
                "tone_adjustment": "upbeat"
            }
        }
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze customer message sentiment."""
        
        result = self.sentiment_analyzer(text)[0]
        label = result['label'].lower()
        score = result['score']
        
        # Map labels to our categories
        if label in ['negative', 'neg']:
            return 'negative', score
        elif label in ['positive', 'pos']:
            return 'positive', score
        else:
            return 'neutral', score
    
    def modify_response_for_sentiment(
        self, 
        response: str, 
        sentiment: str,
        sentiment_confidence: float
    ) -> str:
        """Modify response based on detected sentiment."""
        
        modifier = self.response_modifiers.get(sentiment, self.response_modifiers["neutral"])
        
        # Only apply strong modifications if confidence is high
        if sentiment_confidence < 0.7:
            return response
        
        modified = response
        
        if sentiment == "negative":
            # Add empathy at the beginning
            import random
            empathy = random.choice(modifier["empathy_phrases"])
            if empathy.lower() not in response.lower():
                modified = f"{modifier['prefix']}{empathy} {modifier['urgency']}\n\n{response}"
        
        elif sentiment == "positive":
            # Add positive reinforcement
            modified = f"{modifier['prefix']}{response}"
            if modifier.get('reinforcement') and modifier['reinforcement'].lower() not in response.lower():
                modified += f" {modifier['reinforcement']}"
        
        return modified
```

---

## 3. Integration with External APIs for Dynamic Information Retrieval

### 3.1 Problem Analysis

**Current Limitations:**

| Limitation | Impact |
|------------|--------|
| Static knowledge base | Cannot provide real-time order status |
| No inventory data | Cannot confirm product availability |
| No pricing API | Prices may be outdated |
| No carrier integration | Cannot track shipments in real-time |

### 3.2 Solution: API Integration Layer

#### 3.2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Integration Layer                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  API Registry   │  │  Rate Limiter   │  │  Circuit Breaker│     │
│  │  (Service Map)  │  │  (Per Service)  │  │  (Fault Tolerance)    │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           └────────────────────┼────────────────────┘               │
│                                │                                     │
│                    ┌───────────▼───────────┐                        │
│                    │   API Gateway         │                        │
│                    │   (Request Router)    │                        │
│                    └───────────┬───────────┘                        │
│                                │                                     │
│  ┌─────────────────────────────┼─────────────────────────────┐     │
│  │              │              │              │               │     │
│  ▼              ▼              ▼              ▼               ▼     │
│ ┌────┐       ┌────┐       ┌────┐       ┌────┐       ┌────┐        │
│ │OMS │       │INV │       │SHIP│       │PAY │       │EXT │        │
│ │API │       │API │       │API │       │API │       │API │        │
│ └────┘       └────┘       └────┘       └────┘       └────┘        │
│ Orders       Inventory    Shipping     Payment     External        │
│                                                    (Weather,etc)   │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 API Registry and Configuration

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
import httpx

class APICategory(Enum):
    ORDER_MANAGEMENT = "order_management"
    INVENTORY = "inventory"
    SHIPPING = "shipping"
    PAYMENT = "payment"
    EXTERNAL = "external"

@dataclass
class APIEndpoint:
    name: str
    category: APICategory
    base_url: str
    auth_type: str  # "api_key", "oauth2", "basic"
    auth_header: str
    timeout: float = 5.0
    retry_count: int = 3
    rate_limit: int = 100  # requests per minute
    
@dataclass
class APIRegistry:
    """Central registry for all external APIs."""
    
    endpoints: Dict[str, APIEndpoint] = field(default_factory=dict)
    
    def register(self, endpoint: APIEndpoint) -> None:
        self.endpoints[endpoint.name] = endpoint
    
    def get(self, name: str) -> Optional[APIEndpoint]:
        return self.endpoints.get(name)
    
    def get_by_category(self, category: APICategory) -> List[APIEndpoint]:
        return [ep for ep in self.endpoints.values() if ep.category == category]

# Initialize registry with TechMart APIs
API_REGISTRY = APIRegistry()

# Order Management System
API_REGISTRY.register(APIEndpoint(
    name="techmart_oms",
    category=APICategory.ORDER_MANAGEMENT,
    base_url="https://api.techmart.com/v1/orders",
    auth_type="api_key",
    auth_header="X-API-Key",
    timeout=5.0,
    rate_limit=200
))

# Inventory System
API_REGISTRY.register(APIEndpoint(
    name="techmart_inventory",
    category=APICategory.INVENTORY,
    base_url="https://api.techmart.com/v1/inventory",
    auth_type="api_key",
    auth_header="X-API-Key",
    timeout=3.0,
    rate_limit=500
))

# Shipping Carriers
API_REGISTRY.register(APIEndpoint(
    name="ups_tracking",
    category=APICategory.SHIPPING,
    base_url="https://onlinetools.ups.com/track/v1",
    auth_type="oauth2",
    auth_header="Authorization",
    timeout=10.0,
    rate_limit=50
))

API_REGISTRY.register(APIEndpoint(
    name="fedex_tracking",
    category=APICategory.SHIPPING,
    base_url="https://apis.fedex.com/track/v1",
    auth_type="oauth2",
    auth_header="Authorization",
    timeout=10.0,
    rate_limit=50
))
```

#### 3.2.3 API Client with Resilience Patterns

```python
import asyncio
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Implements circuit breaker pattern for API calls."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.half_open_successes = 0
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if recovery timeout has passed
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = "half-open"
                self.half_open_successes = 0
                return True
            return False
        
        if self.state == "half-open":
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record successful request."""
        
        if self.state == "half-open":
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_requests:
                self.state = "closed"
                self.failures = 0
        else:
            self.failures = 0
    
    def record_failure(self) -> None:
        """Record failed request."""
        
        self.failures += 1
        self.last_failure_time = datetime.now()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failures} failures")


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = datetime.now()
    
    async def acquire(self) -> bool:
        """Acquire a token, waiting if necessary."""
        
        self._refill()
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        
        # Wait for token
        wait_time = 60 / self.rate
        await asyncio.sleep(wait_time)
        self._refill()
        self.tokens -= 1
        return True
    
    def _refill(self) -> None:
        """Refill tokens based on time passed."""
        
        now = datetime.now()
        time_passed = (now - self.last_update).total_seconds()
        self.tokens = min(self.rate, self.tokens + time_passed * (self.rate / 60))
        self.last_update = now


class ResilientAPIClient:
    """API client with circuit breaker, rate limiting, and retry logic."""
    
    def __init__(self, endpoint: APIEndpoint, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(endpoint.rate_limit)
        self.client = httpx.AsyncClient(timeout=endpoint.timeout)
    
    async def request(
        self, 
        method: str, 
        path: str, 
        params: Dict = None,
        json_data: Dict = None
    ) -> Dict:
        """Make resilient API request."""
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise APIUnavailableError(f"{self.endpoint.name} circuit breaker is open")
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Build request
        url = f"{self.endpoint.base_url}/{path}"
        headers = {self.endpoint.auth_header: self.api_key}
        
        # Retry logic
        last_exception = None
        for attempt in range(self.endpoint.retry_count):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=headers
                )
                response.raise_for_status()
                self.circuit_breaker.record_success()
                return response.json()
                
            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_exception = e
                    logger.warning(f"Server error on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    # Client error, don't retry
                    raise APIClientError(str(e))
        
        # All retries failed
        self.circuit_breaker.record_failure()
        raise APIUnavailableError(f"All retries failed: {last_exception}")
    
    async def get(self, path: str, params: Dict = None) -> Dict:
        return await self.request("GET", path, params=params)
    
    async def post(self, path: str, json_data: Dict = None) -> Dict:
        return await self.request("POST", path, json_data=json_data)


class APIUnavailableError(Exception):
    """Raised when API is unavailable."""
    pass

class APIClientError(Exception):
    """Raised for client-side errors (4xx)."""
    pass
```

#### 3.2.4 Service-Specific Integrations

```python
from typing import Optional
from datetime import datetime

class OrderManagementService:
    """Integration with Order Management System."""
    
    def __init__(self, api_client: ResilientAPIClient):
        self.client = api_client
    
    async def get_order(self, order_id: str) -> Dict:
        """Get order details by ID."""
        
        try:
            return await self.client.get(f"orders/{order_id}")
        except APIUnavailableError:
            return self._get_fallback_response("order_status")
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Get order status with tracking info."""
        
        try:
            order = await self.client.get(f"orders/{order_id}/status")
            
            # Enrich with tracking if available
            if tracking := order.get("tracking_number"):
                carrier = order.get("carrier", "ups")
                tracking_info = await self._get_tracking(tracking, carrier)
                order["tracking_details"] = tracking_info
            
            return order
        except APIUnavailableError:
            return self._get_fallback_response("order_status")
    
    async def _get_tracking(self, tracking_number: str, carrier: str) -> Dict:
        """Get tracking details from carrier."""
        
        # This would call the appropriate carrier API
        pass
    
    def _get_fallback_response(self, query_type: str) -> Dict:
        """Return graceful fallback when API unavailable."""
        
        return {
            "status": "unavailable",
            "fallback": True,
            "message": "Real-time order information is temporarily unavailable. "
                      "Please check your confirmation email or contact support.",
            "support_email": "support@techmart.com",
            "support_phone": "1-800-TECHMART"
        }
    
    def format_for_chat(self, order_data: Dict) -> str:
        """Format order data for chat response."""
        
        if order_data.get("fallback"):
            return order_data["message"]
        
        status_emoji = {
            "processing": "⏳",
            "shipped": "📦",
            "in_transit": "🚚",
            "delivered": "✅",
            "cancelled": "❌"
        }
        
        status = order_data.get("status", "unknown")
        emoji = status_emoji.get(status, "📋")
        
        response = f"""
{emoji} **Order #{order_data.get('order_id')}**

**Status:** {status.replace('_', ' ').title()}
**Ordered:** {order_data.get('order_date')}
"""
        
        if tracking := order_data.get("tracking_details"):
            response += f"""
**Tracking:** {order_data.get('tracking_number')}
**Carrier:** {order_data.get('carrier', 'N/A').upper()}
**Location:** {tracking.get('current_location', 'In transit')}
**Expected Delivery:** {tracking.get('expected_delivery', 'Check carrier website')}
"""
        
        return response.strip()


class InventoryService:
    """Integration with Inventory Management System."""
    
    def __init__(self, api_client: ResilientAPIClient):
        self.client = api_client
        self.cache = {}  # Simple cache for product availability
        self.cache_ttl = 300  # 5 minutes
    
    async def check_availability(
        self, 
        product_id: str, 
        location: str = None
    ) -> Dict:
        """Check product availability."""
        
        cache_key = f"{product_id}:{location}"
        
        # Check cache
        if cached := self._get_from_cache(cache_key):
            return cached
        
        try:
            params = {"location": location} if location else {}
            result = await self.client.get(f"products/{product_id}/availability", params)
            
            # Cache result
            self._set_cache(cache_key, result)
            
            return result
        except APIUnavailableError:
            return {"available": "unknown", "fallback": True}
    
    async def get_pricing(self, product_id: str) -> Dict:
        """Get current product pricing."""
        
        try:
            return await self.client.get(f"products/{product_id}/pricing")
        except APIUnavailableError:
            return {"price": "unavailable", "fallback": True}
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Dict) -> None:
        self.cache[key] = (data, datetime.now())
    
    def format_for_chat(self, availability_data: Dict, product_name: str) -> str:
        """Format availability for chat response."""
        
        if availability_data.get("fallback"):
            return f"I couldn't check real-time availability for {product_name}. Please check our website or contact support."
        
        in_stock = availability_data.get("in_stock", False)
        quantity = availability_data.get("quantity", 0)
        
        if in_stock:
            response = f"✅ **{product_name}** is in stock!"
            if quantity <= 5:
                response += f"\n⚠️ Only {quantity} left - order soon!"
        else:
            restock_date = availability_data.get("restock_date")
            response = f"❌ **{product_name}** is currently out of stock."
            if restock_date:
                response += f"\n📅 Expected back in stock: {restock_date}"
        
        return response
```

#### 3.2.5 Integration with RAG Pipeline

```python
class EnhancedRAGPipeline:
    """RAG pipeline with external API integration."""
    
    def __init__(
        self,
        vectorstore,
        llm,
        order_service: OrderManagementService,
        inventory_service: InventoryService
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.order_service = order_service
        self.inventory_service = inventory_service
        
        # Entity extractors for API routing
        self.entity_patterns = {
            "order_id": r"(?:order\s*#?\s*|#)(TM-?\d{4}-?\d{6}|\d{6,})",
            "tracking_number": r"(1Z[A-Z0-9]{16}|[0-9]{20,22})",
            "product_id": r"(?:product\s*#?\s*|SKU:\s*)([A-Z]{2,3}-\d{4,})"
        }
    
    async def generate_response(
        self,
        query: str,
        session_id: str,
        conversation_history: List[Dict]
    ) -> Dict:
        """Generate response with API-enriched context."""
        
        # Step 1: Extract entities that might need API calls
        entities = self._extract_entities(query)
        
        # Step 2: Fetch real-time data if entities found
        api_context = await self._fetch_api_context(entities)
        
        # Step 3: Get RAG context from knowledge base
        rag_context = self._get_rag_context(query)
        
        # Step 4: Combine contexts
        combined_context = self._combine_contexts(
            rag_context, 
            api_context,
            conversation_history
        )
        
        # Step 5: Generate response
        response = await self._generate_with_context(query, combined_context)
        
        return {
            "response": response,
            "sources": {
                "rag": bool(rag_context),
                "api": list(api_context.keys()) if api_context else []
            }
        }
    
    def _extract_entities(self, query: str) -> Dict[str, str]:
        """Extract entities that require API calls."""
        
        import re
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            if match := re.search(pattern, query, re.IGNORECASE):
                entities[entity_type] = match.group(1)
        
        return entities
    
    async def _fetch_api_context(self, entities: Dict[str, str]) -> Dict:
        """Fetch real-time data based on extracted entities."""
        
        context = {}
        
        if order_id := entities.get("order_id"):
            order_data = await self.order_service.get_order_status(order_id)
            context["order"] = self.order_service.format_for_chat(order_data)
        
        if product_id := entities.get("product_id"):
            availability = await self.inventory_service.check_availability(product_id)
            context["availability"] = self.inventory_service.format_for_chat(
                availability, 
                product_id
            )
        
        return context
    
    def _get_rag_context(self, query: str) -> str:
        """Retrieve relevant documents from knowledge base."""
        
        docs = self.vectorstore.similarity_search(query, k=4)
        return "\n\n".join([doc.page_content for doc in docs])
    
    def _combine_contexts(
        self,
        rag_context: str,
        api_context: Dict,
        history: List[Dict]
    ) -> str:
        """Combine all context sources into prompt."""
        
        context_parts = []
        
        # Real-time data (highest priority)
        if api_context:
            context_parts.append("## Real-Time Information")
            for key, value in api_context.items():
                context_parts.append(f"### {key.title()}\n{value}")
        
        # Knowledge base
        if rag_context:
            context_parts.append("## Knowledge Base Information")
            context_parts.append(rag_context)
        
        # Conversation history
        if history:
            context_parts.append("## Recent Conversation")
            for turn in history[-3:]:
                context_parts.append(f"User: {turn['user']}")
                context_parts.append(f"Assistant: {turn['assistant']}")
        
        return "\n\n".join(context_parts)
```

---

## 4. Detailed Implementation Examples

### 4.1 Complete Intent Recognition Flow

```python
# Example: Full flow for handling "Where's my order TM-2024-001234?"

async def handle_order_query(query: str, session_id: str):
    # 1. Intent Classification
    intent_result = intent_classifier.predict(query)
    # Result: IntentResult(intent="order_management", sub_intent="order_status", confidence=0.95)
    
    # 2. Entity Extraction
    entities = entity_extractor.extract(query)
    # Result: {"order_id": "TM-2024-001234"}
    
    # 3. Check Ambiguity
    ambiguity = ambiguity_detector.analyze(query, history=[])
    # Result: AmbiguityAnalysis(is_ambiguous=False)
    
    # 4. Fetch Real-Time Data
    order_data = await order_service.get_order_status("TM-2024-001234")
    # Result: {"status": "in_transit", "tracking": "1Z999AA1...", ...}
    
    # 5. Generate Response
    response = await rag_pipeline.generate_response(
        query=query,
        api_context={"order": order_data},
        session_id=session_id
    )
    
    # 6. Post-Process for Consistency
    final_response = post_processor.process(
        response=response,
        query_type="order_status",
        is_first_message=False,
        customer_sentiment="neutral"
    )
    
    return final_response

# Expected Output:
"""
📦 **Order #TM-2024-001234**

Great news! Your order is on its way! Here's the current status:

**Status:** In Transit
**Carrier:** UPS
**Tracking:** 1Z999AA10123456784
**Expected Delivery:** December 15, 2025

You can track your package directly at ups.com using the tracking number above.

Is there anything else I can help you with?
"""
```

### 4.2 Handling Ambiguous Query Flow

```python
# Example: Flow for handling "I need help with my thing"

async def handle_ambiguous_query(query: str, session_id: str):
    # 1. Intent Classification
    intent_result = intent_classifier.predict(query)
    # Result: IntentResult(intent="general_inquiry", confidence=0.35)
    
    # 2. Ambiguity Detection
    ambiguity = ambiguity_detector.analyze(query, history=[])
    # Result: AmbiguityAnalysis(
    #     is_ambiguous=True,
    #     ambiguity_type="vague",
    #     clarification_needed=True
    # )
    
    # 3. Generate Clarification
    clarification = clarification_manager.create_clarification_request(
        session_id=session_id,
        ambiguity=ambiguity,
        original_query=query
    )
    
    return clarification

# Expected Output:
"""
I'd love to help! What would you like assistance with?

• 📦 Product information
• 🚚 Order or shipping status
• ↩️ Returns or refunds
• 🔧 Technical support
• 💬 Something else

Just let me know and I'll do my best to assist!
"""
```

---

## 5. Evaluation Metrics

### 5.1 Intent Recognition Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Overall correct intent classification | >90% |
| **Precision** | True positives / (True + False positives) | >88% |
| **Recall** | True positives / (True + False negatives) | >85% |
| **F1 Score** | Harmonic mean of precision/recall | >86% |
| **Clarification Rate** | % of queries requiring clarification | <15% |

### 5.2 Personality Consistency Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Tone Consistency Score** | Variance in tone across responses | <0.1 |
| **Format Compliance** | % of responses following format rules | >95% |
| **Empathy Score** | Rating of empathy in negative sentiment cases | >4.0/5 |
| **Response Length Variance** | Standard deviation of response lengths | <50 chars |

### 5.3 API Integration Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **API Availability** | Uptime of external APIs | >99.5% |
| **Response Time (p95)** | 95th percentile response time | <3 sec |
| **Fallback Rate** | % of queries using fallback responses | <5% |
| **Data Freshness** | Age of cached data | <5 min |

---

## 6. References

### Academic Papers

1. **Intent Recognition:**
   - Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL-HLT 2019*.
   - Chen, Q., et al. (2019). "BERT for Joint Intent Classification and Slot Filling." *arXiv:1902.10909*.

2. **Dialogue Systems:**
   - Feng, S., et al. (2020). "doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset." *EMNLP 2020*.
   - Budzianowski, P., et al. (2018). "MultiWOZ: A Large-Scale Multi-Domain Wizard-of-Oz Dataset." *EMNLP 2018*.

3. **Sentiment Analysis:**
   - Barbieri, F., et al. (2020). "TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification." *EMNLP 2020*.

4. **RAG Systems:**
   - Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.

### Technical Resources

5. **LangChain Documentation:** https://python.langchain.com/
6. **spaCy NER:** https://spacy.io/usage/linguistic-features#named-entities
7. **HuggingFace Transformers:** https://huggingface.co/docs/transformers/
8. **Circuit Breaker Pattern:** https://martinfowler.com/bliki/CircuitBreaker.html

---

*Document Version: 2.0*  
*Last Updated: December 2025*  
*Course: Natural Language Processing - BITS Pilani*
