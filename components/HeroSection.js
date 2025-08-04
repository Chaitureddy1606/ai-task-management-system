import React from 'react';

export default function HeroSection(props) {
    return (
        <div style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '40px 20px', borderRadius: '10px', marginBottom: '30px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)', textAlign: 'center' }}>
            <h1 style={{ color: '#2e4053', fontSize: '2.5rem', fontWeight: 'bold' }}>🧠 AI Task Management System</h1>
            <p style={{ color: '#7f8c8d', fontSize: '1.2rem' }}>Intelligent Task Assignment & Analytics Dashboard</p>
            <div>
                <span style={{ backgroundColor: '#28a745', color: 'white', padding: '5px 15px', borderRadius: '20px', fontSize: '0.9rem', fontWeight: 'bold' }}>🟢 System Online</span>
                <span style={{ color: '#7f8c8d', fontSize: '0.9rem', margin: '0 10px' }}> | </span>
                <span style={{ color: '#7f8c8d', fontSize: '0.9rem' }}>👥 {props.employeeCount} Team Members</span>
                <span style={{ color: '#7f8c8d', fontSize: '0.9rem', margin: '0 10px' }}> | </span>
                <span style={{ color: '#7f8c8d', fontSize: '0.9rem' }}>🤖 AI Models Loaded</span>
            </div>
        </div>
    );
}
HeroSection.defaultProps = { employeeCount: 0 }; // Example prop for dynamic data 