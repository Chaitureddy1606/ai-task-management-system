import React from 'react';
import PropTypes from 'prop-types';

export default function TabsSection(props) {
    return (
        <div>
            {props.children}  // Render the tab children passed from Dash
        </div>
    );
}
TabsSection.propTypes = {
    children: PropTypes.node  // Allows passing tab content as children
}; 