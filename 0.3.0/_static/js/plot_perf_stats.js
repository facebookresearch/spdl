/**
 * Plot queue statistics using Plotly.js
 *
 * @param {string} divIdPrefix - Prefix for div IDs (e.g., 'config1')
 * @param {Object} data - JSON data containing queue statistics
 * @param {Array<string>} plotTypes - Array of plot types to create (optional, plots all if not provided)
 * @param {boolean} hideSourceSink - If true, hide source and sink series (default: false)
 */
function plotQueueStats(divIdPrefix, data, plotTypes, hideSourceSink = false) {
    // Extract all queue names from the data
    const queueNames = new Set();
    const taskNames = new Set();

    for (const key in data) {
        if (key.endsWith('_timestamp')) {
            const name = key.slice(0, -10); // Remove '_timestamp'
            // Check if this has queue stats (qps, occupancy_rate, etc.)
            if (data[name + '_qps']) {
                queueNames.add(name);
            }
            // Check if this has task stats (ave_time, num_tasks, etc.)
            if (data[name + '_ave_time']) {
                taskNames.add(name);
            }
        }
    }

    if (queueNames.size === 0 && taskNames.size === 0) {
        console.error('No queue or task data found in:', data);
        return;
    }

    const sortedQueues = Array.from(queueNames).sort();
    const sortedTasks = Array.from(taskNames).sort();

    // Helper function to clean legend names by removing pipeline ID
    // Format: 'PIPELINE_ID:STAGE_ID:STAGE_NAME' -> 'STAGE_ID:STAGE_NAME'
    function cleanLegendName(name) {
        const parts = name.split(':');
        if (parts.length > 1) {
            // Remove first element (pipeline ID) and rejoin
            return parts.slice(1).join(' - ');
        }
        return name;
    }

    // Professional color palette
    const colors = [
        '#4C72B0',
        '#DD8452',
        '#55A868',
        '#C44E52',
        '#8172B3',
        '#937860',
        '#DA8BC3',
        '#8C8C8C',
        '#CCB974',
        '#64B5CD',
    ];

    // Configuration for each subplot
    const allSubplots = [
        {
            type: 'qps',
            divId: divIdPrefix + '_qps',
            dataKey: '_qps',
            title: 'Queue Throughput (QPS)',
            ylabel: 'Items/sec',
            yFormat: '.2f',
            dataSource: 'queue',
        },
        {
            type: 'occupancy',
            divId: divIdPrefix + '_occupancy',
            dataKey: '_occupancy_rate',
            title: 'Queue Occupancy',
            ylabel: 'Occupancy (%)',
            yFormat: '.1f',
            multiplier: 100,  // Convert from rate to percentage
            dataSource: 'queue',
        },
        {
            type: 'put_time',
            divId: divIdPrefix + '_put_time',
            dataKey: '_ave_put_time',
            title: 'Average Put Wait Time',
            ylabel: 'Time (ms)',
            yFormat: '.2f',
            multiplier: 1000,  // Convert from seconds to milliseconds
            dataSource: 'queue',
        },
        {
            type: 'get_time',
            divId: divIdPrefix + '_get_time',
            dataKey: '_ave_get_time',
            title: 'Average Get Wait Time',
            ylabel: 'Time (ms)',
            yFormat: '.2f',
            multiplier: 1000,  // Convert from seconds to milliseconds
            dataSource: 'queue',
        },
        {
            type: 'items',
            divId: divIdPrefix + '_items',
            dataKey: '_num_items',
            title: 'Items Processed',
            ylabel: 'Count',
            yFormat: 'd',
            dataSource: 'queue',
        },
        {
            type: 'task_time',
            divId: divIdPrefix + '_task_time',
            dataKey: '_ave_time',
            title: 'Average Task Execution Time',
            ylabel: 'Time (ms)',
            yFormat: '.2f',
            multiplier: 1000,  // Convert from seconds to milliseconds
            dataSource: 'task',
        },
    ];

    // Filter subplots based on plotTypes parameter (if provided) or check div existence
    const subplots = plotTypes
        ? allSubplots.filter(config => plotTypes.includes(config.type))
        : allSubplots.filter(config => document.getElementById(config.divId) !== null);

    // Create each subplot
    subplots.forEach((config, idx) => {
        const plotData = [];

        // Determine which data source to use
        const dataSource = config.dataSource === 'task' ? sortedTasks : sortedQueues;

        dataSource.forEach((name, colorIdx) => {
            const timestamps = data[name + '_timestamp'];
            const values = data[name + config.dataKey];

            if (!timestamps || !values) {
                return;
            }

            // Convert timestamps to Date objects
            const x = timestamps.map(t => new Date(t * 1000));

            // Apply multiplier if specified (e.g., for percentage or milliseconds)
            const y = config.multiplier
                ? values.map(v => v * config.multiplier)
                : values;

            // Clean the name for legend display
            const displayName = cleanLegendName(name);

            // Determine if this is a source or sink series
            const isSourceOrSink = name.includes('src_queue') || name.includes('sink_queue');

            // Create trace object
            const trace = {
                x: x,
                y: y,
                name: displayName,
                type: 'scatter',
                mode: 'lines+markers',
                line: {
                    width: 2,
                    color: colors[colorIdx % colors.length],
                },
                marker: {
                    size: 4,
                    color: colors[colorIdx % colors.length],
                },
                hovertemplate: '<b>%{fullData.name}</b><br>' +
                                'Time: %{x}<br>' +
                                config.ylabel + ': %{y:' + config.yFormat + '}' +
                                '<extra></extra>',
            };

            // Hide source/sink if option is enabled
            if (hideSourceSink && isSourceOrSink) {
                trace.visible = 'legendonly';
            }

            plotData.push(trace);
        });

        const layout = {
            title: {
                text: config.title,
                font: {
                    size: 14,
                    weight: 'bold',
                },
            },
            xaxis: {
                title: idx === subplots.length - 1 ? 'Time' : '',
                type: 'date',
                tickformat: '%H:%M:%S',
            },
            yaxis: {
                title: config.ylabel,
                rangemode: config.type === 'occupancy' ? undefined : 'tozero',
                range: config.type === 'occupancy' ? [0, 105] : undefined,
            },
            showlegend: true,  // Show legend on all subplots
            legend: {
                orientation: 'v',
                x: 1.02,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
            },
            hovermode: 'closest',
            height: 220,  // 60% of default height (450px * 0.6 = 270px)
            margin: {
                l: 80,
                r: 150,
                t: 50,
                b: 50,
            },
        };

        const plotConfig = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
        };

        try {
            Plotly.newPlot(config.divId, plotData, layout, plotConfig);
        } catch (error) {
            const div = document.getElementById(config.divId);
            if (div) {
                div.textContent = 'Failed to plot: ' + error.toString();
            }
            console.error('Plotting error for', config.divId, ':', error);
        }
    });
}

/**
 * Load and plot queue statistics from a JSON file with dynamic DOM creation
 *
 * @param {string} parentDivId - ID of the parent div element where plots will be inserted
 * @param {string} dataUrl - URL to the JSON data file
 * @param {Array<string>} plotTypes - Array of plot types to create
 *                                    Valid types: 'qps', 'occupancy', 'put_time', 'get_time', 'items', 'task_time'
 * @param {boolean} hideSourceSink - If true, hide source and sink series (default: false)
 */
function loadAndPlotQueueStats(parentDivId, dataUrl, plotTypes, hideSourceSink = false) {
    const parentDiv = document.getElementById(parentDivId);
    if (!parentDiv) {
        console.error('Parent div not found:', parentDivId);
        return;
    }

    // Map of plot type to subplot configuration
    const plotTypeMap = {
        'qps': { suffix: '_qps', name: 'qps' },
        'occupancy': { suffix: '_occupancy', name: 'occupancy' },
        'put_time': { suffix: '_put_time', name: 'put_time' },
        'get_time': { suffix: '_get_time', name: 'get_time' },
        'items': { suffix: '_items', name: 'items' },
        'task_time': { suffix: '_task_time', name: 'task_time' },
    };

    // Validate plot types
    const validPlotTypes = plotTypes.filter(type => {
        if (!plotTypeMap[type]) {
            console.warn('Invalid plot type:', type);
            return false;
        }
        return true;
    });

    if (validPlotTypes.length === 0) {
        console.error('No valid plot types specified');
        parentDiv.textContent = 'Error: No valid plot types specified';
        return;
    }

    // Generate unique prefix for this set of plots
    const divIdPrefix = parentDivId + '_plot';

    // Create div elements for each plot type
    validPlotTypes.forEach(type => {
        const plotDiv = document.createElement('div');
        plotDiv.id = divIdPrefix + '_' + type;
        plotDiv.style.marginBottom = '20px';
        parentDiv.appendChild(plotDiv);
    });

    // Fetch data and plot
    fetch(dataUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch data from: ' + dataUrl);
            }
            return response.json();
        })
        .then(data => {
            plotQueueStats(divIdPrefix, data, validPlotTypes, hideSourceSink);
        })
        .catch(error => {
            console.error('Error loading queue stats:', error);
            parentDiv.innerHTML = '<div style="color: red; padding: 10px;">Error loading data: ' +
                                    error.toString() + '</div>';
        });
}
