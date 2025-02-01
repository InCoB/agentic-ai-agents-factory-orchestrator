# AI Agent Factory - Gradio Interface Documentation

## Overview
The AI Agent Factory Gradio interface provides a user-friendly web interface for interacting with our AI agent system. It allows users to input prompts, choose between dynamic and fixed workflows, and receive detailed analysis and logs.

## Features

### 1. Input Components
- **User Prompt**: A text input area where users can enter their queries or requests
  - Lines: 3
  - Placeholder: "Enter your prompt here (e.g., 'Find me a stock to invest right now')"
  - Info: "Be specific about what you want the AI agents to do."

- **Dynamic Workflow Toggle**: A checkbox to control workflow generation
  - Default: Enabled (True)
  - Info: "If checked, the system will generate a custom workflow for your prompt."

### 2. Control Buttons
- **Run Analysis** (🚀): Executes the AI workflow with the current prompt
- **Clear** (🧹): Resets both the output and logs

### 3. Output Components
- **Analysis Output**: Displays the final results
  - Lines: 10
  - Copy button: Enabled
  - Format: Markdown-compatible

- **Detailed Logs**: Collapsible section showing system logs
  - Lines: 15
  - Copy button: Enabled
  - Initially: Collapsed

### 4. Example Prompts
Pre-configured examples that users can try:
1. "Find me a stock to invest right now"
2. "Analyze the latest tech industry trends"
3. "What are the best performing stocks in the renewable energy sector?"

## Usage Instructions

1. **Basic Usage**:
   - Enter your prompt in the text input area
   - Choose whether to use dynamic workflow
   - Click "🚀 Run Analysis"
   - View results in the Analysis Output section
   - Expand "Detailed Logs" for process information

2. **Using Examples**:
   - Click any example prompt to automatically populate the input
   - Examples are cached for faster execution
   - Each example demonstrates different capabilities

3. **Clearing Output**:
   - Use the "🧹 Clear" button to reset both output and logs
   - This is useful when starting a new analysis

## Technical Details

### Server Configuration
- Local URL: http://127.0.0.1:7860
- Public sharing: Enabled
- Debug mode: Enabled
- Authentication: None required
- Auto-browser launch: Enabled

### Performance Features
- Example caching for improved response time
- Asynchronous processing for better performance
- Copy buttons for easy result sharing
- Markdown support in output

## Error Handling
- Clear error messages displayed in debug mode
- Detailed logging of process steps
- Graceful handling of API failures
- Input validation and sanitization

## Best Practices
1. Be specific in your prompts for better results
2. Check the logs for detailed process information
3. Use the dynamic workflow for complex queries
4. Try the examples to understand system capabilities

## Limitations
- Processing time depends on query complexity
- API rate limits may affect response time
- Some features require internet connectivity
- Output format is text-based (no direct file downloads)

## Troubleshooting
1. If the interface doesn't load:
   - Check if the server is running
   - Verify the correct URL (http://127.0.0.1:7860)
   - Ensure no other service is using port 7860

2. If analysis fails:
   - Check the logs for error messages
   - Verify internet connectivity
   - Ensure the prompt is properly formatted

3. If examples don't work:
   - Clear the cache directory
   - Restart the application
   - Check API connectivity 