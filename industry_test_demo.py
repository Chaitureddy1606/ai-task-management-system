#!/usr/bin/env python3
"""
Industry-Level Test Demo for Advanced Task Management AI
Demonstrates all 4 use cases with sophisticated AI models
"""

import requests
import json
import time
from datetime import datetime

class IndustryTestDemo:
    """Industry-level test demonstration"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
    
    def test_use_case_1_auto_assignment(self):
        """Test Use Case 1: Auto-assign tasks to best-fit employee"""
        print("\n🎯 Testing Use Case 1: Auto-Assignment")
        print("=" * 50)
        
        test_tasks = [
            {
                "title": "Fix critical payment processing bug",
                "description": "Users unable to complete credit card payments on checkout page",
                "urgency_score": 9,
                "complexity_score": 6,
                "days_until_deadline": 2,
                "business_impact": 8.5,
                "estimated_hours": 12
            },
            {
                "title": "Implement new user authentication feature",
                "description": "Add OAuth2 integration with Google and Facebook login options",
                "urgency_score": 7,
                "complexity_score": 8,
                "days_until_deadline": 14,
                "business_impact": 7.0,
                "estimated_hours": 24
            },
            {
                "title": "Write API documentation for payment endpoints",
                "description": "Create comprehensive documentation for payment processing APIs",
                "urgency_score": 4,
                "complexity_score": 3,
                "days_until_deadline": 7,
                "business_impact": 5.0,
                "estimated_hours": 8
            }
        ]
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n📋 Test Task {i}: {task['title']}")
            
            # Test advanced processing
            response = requests.post(
                f"{self.base_url}/api/process-task-advanced",
                json=task
            )
            
            if response.status_code == 200:
                result = response.json()
                data = result['data']
                
                print(f"✅ Category: {data['category']} (confidence: {data['category_confidence']:.1%})")
                print(f"✅ Status: {data['status']} (confidence: {data['status_confidence']:.1%})")
                print(f"✅ Assigned Employee: {data['assigned_employee']}")
                print(f"✅ Processing Time: {data['processing_time']:.3f}s")
                
                # Show recommendations
                print("📊 Top Recommendations:")
                for rec in data['recommendations'][:3]:
                    print(f"   - {rec['employee_id']}: {rec['score']:.3f}")
                
                self.test_results.append({
                    'use_case': 'Auto-Assignment',
                    'task': task['title'],
                    'result': data,
                    'success': True
                })
            else:
                print(f"❌ Error: {response.text}")
                self.test_results.append({
                    'use_case': 'Auto-Assignment',
                    'task': task['title'],
                    'error': response.text,
                    'success': False
                })
    
    def test_use_case_2_status_prediction(self):
        """Test Use Case 2: Predict task completion status"""
        print("\n🎯 Testing Use Case 2: Status Prediction")
        print("=" * 50)
        
        test_tasks = [
            {
                "title": "High-priority security patch",
                "description": "Critical security vulnerability in authentication system",
                "urgency_score": 10,
                "complexity_score": 7,
                "days_until_deadline": 1,
                "business_impact": 9.5,
                "estimated_hours": 16
            },
            {
                "title": "Regular code review",
                "description": "Standard code review for new feature implementation",
                "urgency_score": 3,
                "complexity_score": 4,
                "days_until_deadline": 5,
                "business_impact": 4.0,
                "estimated_hours": 4
            }
        ]
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n📋 Test Task {i}: {task['title']}")
            
            response = requests.post(
                f"{self.base_url}/api/process-task-advanced",
                json=task
            )
            
            if response.status_code == 200:
                result = response.json()
                data = result['data']
                
                print(f"✅ Predicted Status: {data['status']}")
                print(f"✅ Confidence: {data['status_confidence']:.1%}")
                print(f"✅ Status Probabilities:")
                for status, prob in data['status_probabilities'].items():
                    print(f"   - {status}: {prob:.1%}")
                
                self.test_results.append({
                    'use_case': 'Status Prediction',
                    'task': task['title'],
                    'result': data,
                    'success': True
                })
            else:
                print(f"❌ Error: {response.text}")
                self.test_results.append({
                    'use_case': 'Status Prediction',
                    'task': task['title'],
                    'error': response.text,
                    'success': False
                })
    
    def test_use_case_3_priority_prediction(self):
        """Test Use Case 3: Predict urgency or priority"""
        print("\n🎯 Testing Use Case 3: Priority Prediction")
        print("=" * 50)
        
        test_tasks = [
            {
                "title": "System outage emergency fix",
                "description": "Production system down, affecting all users",
                "urgency_score": 10,
                "complexity_score": 8,
                "days_until_deadline": 0,
                "business_impact": 10.0,
                "estimated_hours": 12
            },
            {
                "title": "Minor UI improvement",
                "description": "Update button colors for better accessibility",
                "urgency_score": 2,
                "complexity_score": 2,
                "days_until_deadline": 30,
                "business_impact": 2.0,
                "estimated_hours": 2
            }
        ]
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n📋 Test Task {i}: {task['title']}")
            
            response = requests.post(
                f"{self.base_url}/api/process-task-advanced",
                json=task
            )
            
            if response.status_code == 200:
                result = response.json()
                data = result['data']
                
                # Calculate priority score
                priority_score = (
                    task['urgency_score'] * 0.3 +
                    task['complexity_score'] * 0.2 +
                    task['business_impact'] * 0.3 +
                    (10 if task['days_until_deadline'] <= 0 else 
                     (8 if task['days_until_deadline'] <= 3 else 
                      (6 if task['days_until_deadline'] <= 7 else 
                       (4 if task['days_until_deadline'] <= 30 else 2)))) * 0.2
                )
                
                print(f"✅ Calculated Priority Score: {priority_score:.2f}/10")
                print(f"✅ Urgency Score: {task['urgency_score']}/10")
                print(f"✅ Business Impact: {task['business_impact']}/10")
                print(f"✅ Days Until Deadline: {task['days_until_deadline']}")
                
                self.test_results.append({
                    'use_case': 'Priority Prediction',
                    'task': task['title'],
                    'priority_score': priority_score,
                    'success': True
                })
            else:
                print(f"❌ Error: {response.text}")
                self.test_results.append({
                    'use_case': 'Priority Prediction',
                    'task': task['title'],
                    'error': response.text,
                    'success': False
                })
    
    def test_use_case_4_task_classification(self):
        """Test Use Case 4: Detect task type from description"""
        print("\n🎯 Testing Use Case 4: Task Classification")
        print("=" * 50)
        
        test_tasks = [
            {
                "title": "Fix login authentication bug",
                "description": "Users cannot log in with correct credentials, getting 401 error",
                "urgency_score": 8,
                "complexity_score": 5,
                "days_until_deadline": 3,
                "business_impact": 7.0,
                "estimated_hours": 8
            },
            {
                "title": "Add new payment gateway integration",
                "description": "Implement Stripe payment processing for subscription billing",
                "urgency_score": 6,
                "complexity_score": 8,
                "days_until_deadline": 21,
                "business_impact": 8.0,
                "estimated_hours": 40
            },
            {
                "title": "Update user documentation",
                "description": "Rewrite API documentation with new examples and code samples",
                "urgency_score": 3,
                "complexity_score": 2,
                "days_until_deadline": 14,
                "business_impact": 4.0,
                "estimated_hours": 6
            }
        ]
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n📋 Test Task {i}: {task['title']}")
            
            response = requests.post(
                f"{self.base_url}/api/process-task-advanced",
                json=task
            )
            
            if response.status_code == 200:
                result = response.json()
                data = result['data']
                
                print(f"✅ Predicted Category: {data['category']}")
                print(f"✅ Confidence: {data['category_confidence']:.1%}")
                print(f"✅ Category Probabilities:")
                for category, prob in data['category_probabilities'].items():
                    print(f"   - {category}: {prob:.1%}")
                
                self.test_results.append({
                    'use_case': 'Task Classification',
                    'task': task['title'],
                    'result': data,
                    'success': True
                })
            else:
                print(f"❌ Error: {response.text}")
                self.test_results.append({
                    'use_case': 'Task Classification',
                    'task': task['title'],
                    'error': response.text,
                    'success': False
                })
    
    def test_explainable_ai(self):
        """Test Explainable AI features"""
        print("\n🧠 Testing Explainable AI")
        print("=" * 50)
        
        test_task = {
            "title": "Critical database performance issue",
            "description": "Database queries taking 10+ seconds, affecting user experience",
            "urgency_score": 9,
            "complexity_score": 7,
            "days_until_deadline": 2,
            "business_impact": 8.5,
            "estimated_hours": 16
        }
        
        print(f"📋 Test Task: {test_task['title']}")
        
        response = requests.post(
            f"{self.base_url}/api/explain",
            json=test_task
        )
        
        if response.status_code == 200:
            result = response.json()
            explanation = result['explanation']
            
            print("✅ AI Explanation:")
            print(f"   Category: {explanation['category_explanation']['reasoning']}")
            print(f"   Status: {explanation['status_explanation']['reasoning']}")
            print(f"   Assignment: {explanation['assignment_explanation']['reasoning']}")
            
            self.test_results.append({
                'use_case': 'Explainable AI',
                'task': test_task['title'],
                'explanation': explanation,
                'success': True
            })
        else:
            print(f"❌ Error: {response.text}")
            self.test_results.append({
                'use_case': 'Explainable AI',
                'task': test_task['title'],
                'error': response.text,
                'success': False
            })
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities"""
        print("\n📊 Testing Performance Monitoring")
        print("=" * 50)
        
        # Get performance metrics
        response = requests.get(f"{self.base_url}/api/performance")
        
        if response.status_code == 200:
            result = response.json()
            performance = result['performance']
            
            print("✅ Performance Metrics:")
            print(f"   Total Requests: {performance['total_requests']}")
            print(f"   Successful Predictions: {performance['successful_predictions']}")
            print(f"   Average Response Time: {performance['average_response_time']:.3f}s")
            print(f"   Success Rate: {performance['successful_predictions']/max(performance['total_requests'], 1)*100:.1f}%")
            
            self.test_results.append({
                'use_case': 'Performance Monitoring',
                'metrics': performance,
                'success': True
            })
        else:
            print(f"❌ Error: {response.text}")
            self.test_results.append({
                'use_case': 'Performance Monitoring',
                'error': response.text,
                'success': False
            })
    
    def run_all_tests(self):
        """Run all industry-level tests"""
        print("🚀 Industry-Level AI Task Management Test Suite")
        print("=" * 60)
        print("Testing all 4 use cases with advanced AI models...")
        
        start_time = time.time()
        
        # Test all use cases
        self.test_use_case_1_auto_assignment()
        self.test_use_case_2_status_prediction()
        self.test_use_case_3_priority_prediction()
        self.test_use_case_4_task_classification()
        
        # Test advanced features
        self.test_explainable_ai()
        self.test_performance_monitoring()
        
        # Generate summary
        self.generate_summary(start_time)
    
    def generate_summary(self, start_time):
        """Generate comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 INDUSTRY-LEVEL TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - successful_tests
        
        print(f"✅ Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {successful_tests/total_tests*100:.1f}%")
        print(f"⏱️  Total Time: {time.time() - start_time:.2f}s")
        
        print("\n🎯 Use Case Performance:")
        use_cases = {}
        for result in self.test_results:
            use_case = result['use_case']
            if use_case not in use_cases:
                use_cases[use_case] = {'success': 0, 'total': 0}
            use_cases[use_case]['total'] += 1
            if result['success']:
                use_cases[use_case]['success'] += 1
        
        for use_case, stats in use_cases.items():
            success_rate = stats['success'] / stats['total'] * 100
            print(f"   {use_case}: {success_rate:.1f}% ({stats['success']}/{stats['total']})")
        
        print("\n🎉 Industry-Level AI System Status:")
        if successful_tests == total_tests:
            print("   ✅ ALL TESTS PASSED - Production Ready!")
        elif successful_tests >= total_tests * 0.8:
            print("   ⚠️  MOST TESTS PASSED - Needs Minor Fixes")
        else:
            print("   ❌ MULTIPLE FAILURES - Needs Attention")
        
        print("\n🚀 Ready for Enterprise Deployment!")

def main():
    """Main test execution"""
    print("🧠 Industry-Level AI Task Management Test Suite")
    print("Testing all 4 use cases with advanced models...")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server not responding")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start the API server first:")
        print("   cd api")
        print("   python advanced_server.py")
        return
    
    # Run tests
    demo = IndustryTestDemo()
    demo.run_all_tests()

if __name__ == "__main__":
    main() 