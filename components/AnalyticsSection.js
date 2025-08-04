import React from 'react';
import PropTypes from 'prop-types';

export default function AnalyticsSection(props) {
    return (
        <div style={{ padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '10px' }}>
            <h3 style={{ color: '#2e4053' }}>📊 Comprehensive Analytics Dashboard</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                {props.metrics && props.metrics.map((metric, index) => (
                    <div key={index} style={{ width: '24%', margin: '1%', backgroundColor: '#ffffff', padding: '10px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <h4>{metric.label}</h4>
                        <h2>{metric.value}</h2>
                    </div>
                ))}
            </div>
            <div>
                {props.children}  // For embedding graphs or other dynamic content
            </div>
        </div>
    );
}
AnalyticsSection.propTypes = {
    metrics: PropTypes.arrayOf(PropTypes.shape({
        label: PropTypes.string,
        value: PropTypes.oneOfType([PropTypes.string, PropTypes.number])
    })),
    children: PropTypes.node
};
AnalyticsSection.defaultProps = {
    metrics: []
}; 