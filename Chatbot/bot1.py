import requests
import logging
import os
import subprocess
import json
from datetime import datetime
from collections import Counter
from lead_scorer import (
    judge_lead,
    load_all_saved_models,
    _get_prediction_and_probs,
    rule_score_to_confidence,
    save_lead_to_csv,
)

class LeadScorerBot:
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"
        self.model = "deepseek-r1:1.5b"
        self.last_lead = None
        self.state = 0  # 0: waiting for message, 1: waiting for name, 2: waiting for company, 3: waiting for job, 4: waiting for source, 5: asking to save
        self.pending_lead = {}

    def call_llm(self, prompt):
        try:
            response = requests.post(self.api_url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            })
            if response.status_code == 200:
                return response.json().get("response", "No response")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

    def process_message(self, user_message):
        msg = user_message.lower()

        # Handle special commands
        if "help" in msg:
            return "I can help score leads using a hybrid approach of rule-based scoring and machine learning models. Start by entering the lead message."
        elif "example" in msg or "how to" in msg:
            return "Start by entering the lead message, then I'll ask for additional details step by step."
        elif any(phrase in msg for phrase in ["who are you", "what are you", "who is this", "what is this", "tell me about yourself", "introduce yourself", "what do you do", "who am i talking to"]):
            return "I am your AI assistant for lead scoring. I collect lead information step by step."
        elif "reset" in msg or "start over" in msg:
            self.state = 0
            self.pending_lead = {}
            return "Conversation reset. Please enter the lead message to begin."
        elif "exit" in msg or "bye" in msg or "goodbye" in msg:
            self.state = 0
            self.pending_lead = {}
            return "Goodbye! Thank you for using the lead scoring assistant. Have a great day!"

        # Handle state-based conversation
        if self.state == 0:
            # Waiting for message
            if user_message.strip():
                self.pending_lead['message'] = user_message
                self.state = 1
                return "Got the message. What's the lead's name?"
            else:
                return "Please enter the lead message to begin scoring."

        elif self.state == 1:
            # Waiting for name
            self.pending_lead['name'] = user_message
            self.state = 2
            return "Got the name. What's the company?"

        elif self.state == 2:
            # Waiting for company
            self.pending_lead['company'] = user_message
            self.state = 3
            return "Got the company. What's the job title?"

        elif self.state == 3:
            # Waiting for job title
            self.pending_lead['job_title'] = user_message
            self.state = 4
            return "Got the job title. What's the source? (e.g., email, LinkedIn, contact form)"

        elif self.state == 4:
            # Waiting for source, then score
            self.pending_lead['source'] = user_message

            # Score the lead
            name = self.pending_lead.get('name', 'unknown')
            company = self.pending_lead.get('company', 'unknown')
            job_title = self.pending_lead.get('job_title', 'unknown')
            source = self.pending_lead.get('source', 'unknown')
            message = self.pending_lead.get('message', '')

            # Rule-based scoring
            rule_score, rule_label, rule_reason, _ = judge_lead(message, job_title, source, company)
            rule_confidence = rule_score_to_confidence(rule_score)

            # ML models
            models_loaded = load_all_saved_models()
            ml_results = []
            combined_input = f"{message} {job_title} {company}".strip()
            for m in models_loaded:
                try:
                    pred, top_prob, probs_dict = _get_prediction_and_probs(m['pipeline'], combined_input)
                    ml_results.append({
                        'name': m.get('name', 'model'),
                        'pred': pred,
                        'top_prob': top_prob,
                        'probs': probs_dict
                    })
                except Exception as e:
                    pass

            # Determine consensus and final confidence
            ml_preds = [r['pred'] for r in ml_results if r['pred']]
            ml_confs = [r['top_prob'] for r in ml_results if r['top_prob'] is not None]
            consensus = rule_label
            if ml_preds:
                pred_counts = Counter(ml_preds)
                most_common_ml = pred_counts.most_common(1)[0][0]
                consensus = most_common_ml
                average_ml_conf = sum(ml_confs) / len(ml_confs) if ml_confs else None
            else:
                average_ml_conf = None

            final_confidence = (rule_confidence + average_ml_conf) / 2 if average_ml_conf is not None else rule_confidence

            # Override final label based on final confidence
            final_label = consensus
            if final_confidence < 0.5:
                final_label = 'Cold'
            elif final_confidence > 0.75:
                final_label = 'Hot'

            # Store last lead
            self.last_lead = {'name': name, 'company': company, 'job_title': job_title, 'source': source, 'message': message, 'final_label': final_label}

            # Format response
            response = f"**Lead Scored:** {final_label}\n\n**Rule-Based:** {rule_label} (Confidence: {rule_confidence:.1%})\n**Reason:** {rule_reason}\n\n**ML Models:**\n"
            if ml_results:
                for r in ml_results:
                    response += f"- {r['name']}: {r['pred']}\n"
            else:
                response += "No ML models available.\n"
            response += f"\n**Consensus:** {consensus}\n**Final Confidence:** {final_confidence:.1%}\n**Final Label:** {final_label}\n\nWould you like to save this lead? Reply 'yes' or 'no'."

            # Move to save question state
            self.state = 5

            return response

        elif self.state == 5:
            # Waiting for save decision
            msg = user_message.lower().strip()
            if msg in ['yes', 'y', 'save']:
                # Save the lead
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(script_dir, "..", "Data", "csvfile.csv")
                success = save_lead_to_csv(self.last_lead['name'], self.last_lead['company'], self.last_lead['job_title'], self.last_lead['source'], self.last_lead['message'], self.last_lead['final_label'], csv_path=csv_path)
                if success:
                    save_response = "Lead saved to training data."
                    # Trigger retraining check in background
                    try:
                        subprocess.Popen(["python", "auto_retrain.py"])
                    except Exception:
                        pass  # Ignore if retraining fails to start
                else:
                    save_response = "Failed to save lead to training data. Please ensure the CSV file is not open in another program."
            elif msg in ['no', 'n', 'skip']:
                save_response = "Lead not saved."
            else:
                return "Please reply 'yes' to save or 'no' to skip."

            # Give recommendations based on label
            label = self.last_lead['final_label']
            if label == 'Hot':
                recommendations = "🎯 **Recommendations:**\n- Assign to SDR immediately\n- Schedule discovery call within 24 hours\n- Send premium email template\n- Prioritize follow-up"
            elif label == 'Warm':
                recommendations = "📞 **Recommendations:**\n- Send personalized follow-up email\n- Add to nurture campaign\n- Offer free trial or demo\n- Monitor engagement"
            else:  # Cold
                recommendations = "❄️ **Recommendations:**\n- Add to cold lead list\n- Monitor for future engagement\n- Consider re-engagement campaign in 3-6 months\n- Low priority follow-up"

            # Reset for next lead
            self.state = 0
            self.pending_lead = {}

            return f"{save_response}\n\n{recommendations}\n\nReady for the next lead. Please enter a message to begin scoring."

            # Score the lead
            name = self.pending_lead.get('name', 'unknown')
            company = self.pending_lead.get('company', 'unknown')
            job_title = self.pending_lead.get('job_title', 'unknown')
            source = self.pending_lead.get('source', 'unknown')
            message = self.pending_lead.get('message', '')

            # Rule-based scoring
            rule_score, rule_label, rule_reason, _ = judge_lead(message, job_title, source, company)
            rule_confidence = rule_score_to_confidence(rule_score)

            # ML models
            models_loaded = load_all_saved_models()
            ml_results = []
            combined_input = f"{message} {job_title} {company}".strip()
            for m in models_loaded:
                try:
                    pred, top_prob, probs_dict = _get_prediction_and_probs(m['pipeline'], combined_input)
                    ml_results.append({
                        'name': m.get('name', 'model'),
                        'pred': pred,
                        'top_prob': top_prob,
                        'probs': probs_dict
                    })
                except Exception as e:
                    pass

            # Determine consensus and final confidence
            ml_preds = [r['pred'] for r in ml_results if r['pred']]
            ml_confs = [r['top_prob'] for r in ml_results if r['top_prob'] is not None]
            consensus = rule_label
            if ml_preds:
                pred_counts = Counter(ml_preds)
                most_common_ml = pred_counts.most_common(1)[0][0]
                consensus = most_common_ml
                average_ml_conf = sum(ml_confs) / len(ml_confs) if ml_confs else None
            else:
                average_ml_conf = None

            final_confidence = (rule_confidence + average_ml_conf) / 2 if average_ml_conf is not None else rule_confidence

            # Override final label based on final confidence
            final_label = consensus
            if final_confidence < 0.5:
                final_label = 'Cold'
            elif final_confidence > 0.75:
                final_label = 'Hot'

            # Store last lead
            self.last_lead = {'name': name, 'company': company, 'job_title': job_title, 'source': source, 'message': message, 'final_label': final_label}

            # Format response
            response = f"**Lead Scored:** {final_label}\n\n**Rule-Based:** {rule_label} (Confidence: {rule_confidence:.1%})\n**Reason:** {rule_reason}\n\n**ML Models:**\n"
            if ml_results:
                for r in ml_results:
                    response += f"- {r['name']}: {r['pred']}\n"
            else:
                response += "No ML models available.\n"
            response += f"\n**Consensus:** {consensus}\n**Final Confidence:** {final_confidence:.1%}\n**Final Label:** {final_label}\n\nWould you like to save this lead? Reply 'yes' or 'no'."

            # Move to save question state
            self.state = 5

            return response

        return "I'm not sure how to process that. Type 'help' for assistance or 'reset' to start over."

# Example usage
if __name__ == "__main__":
    bot = LeadScorerBot()
    # Simulate a conversation to input a lead
    messages = [
        "Hello, I need more details about your services and would like to schedule a consultation.",
        "Jane Smith",
        "Tech Solutions Inc",
        "VP of Engineering",
        "LinkedIn",
        "yes"  # Save the lead
    ]
    for msg in messages:
        response = bot.process_message(msg)
        print(f"User: {msg}")
        print(f"Bot: {response}")
        print("---")