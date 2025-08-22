import json
import re

response = {
    "choices": [{
        "message": {
            "content": "```json\n{\n  \"recommendations\": [\n    \"Designed and implemented a RESTful API service that improved system response times by 40%, reducing latency from 500ms to 300ms for over 10,000 daily users.\",\n    \"Optimized database queries for a high-traffic application, reducing query execution time by 60% and cutting server costs by $5,000 annually.\",\n    \"Developed an automated testing framework that reduced regression testing time by 75%, saving the team 20+ hours per sprint.\",\n    \"Led the migration of legacy systems to a microservices architecture, improving scalability and reducing downtime incidents by 90%.\",\n    \"Built a real-time monitoring dashboard that identified system bottlenecks, decreasing incident resolution time by 50%.\",\n    \"Implemented CI/CD pipelines using Jenkins and Docker, reducing deployment times from 2 hours to 15 minutes and increasing release frequency by 300%.\",\n    \"Created a data processing tool that automated manual data entry tasks, reducing human errors by 80% and saving 30+ hours per month.\",\n    \"Enhanced application security by integrating OAuth 2.0, reducing unauthorized access attempts by 95%.\",\n    \"Refactored legacy codebase, improving code maintainability and reducing bug reports by 70% over six months.\",\n    \"Collaborated with cross-functional teams to deliver a feature-rich mobile app, increasing user engagement by 25% within three months.\",\n    \"Designed a caching strategy that reduced API load times by 50%, improving user experience for 50,000+ monthly active users.\",\n    \"Automated server provisioning using Terraform, reducing setup time from 4 hours to 30 minutes per environment.\",\n    \"Developed a machine learning model for fraud detection, reducing false positives by 60% and saving $100,000 annually in potential losses.\",\n    \"Introduced A/B testing for key features, leading to a 15% increase in conversion rates over six months.\",\n    \"Mentored junior engineers, resulting in a 40% improvement in team productivity and faster onboarding for new hires.\"\n  ]\n}\n```"
        }
    }]
}

def extract_json(text):
    match = re.search(r'```json\n([\s\S]*?)\n```', text)
    if not match:
        raise ValueError("No JSON block found")
    return json.loads(match.group(1).strip())

try:
    data = extract_json(response["choices"][0]["message"]["content"])

    print("----List of Recommendations----\n")
    for i, obj in enumerate(data['recommendations']):
        print(f"\nRecommendation {i}: {obj}")
    print()
except (ValueError, json.JSONDecodeError) as e:
    print("Parsing failed:", e)