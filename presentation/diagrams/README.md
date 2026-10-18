# Project architecture diagrams

The Streamlit **How it works** section embeds `prediction-workflow.html` and
`training-evaluation.html`. Each is a standalone interactive page and can also
be opened directly in a browser. Its JSON file is the editable Archify source.

The diagrams describe the Python inference stages and the offline training
experiment. Streamlit and FastAPI are independent entry points into the shared
matcher. Component source links are pinned to the project revision recorded in
each JSON file; they describe that verified model implementation.

Generated with [Archify](https://github.com/tt-a1i/archify), revision
`b25142deac4b19b293e45b6bf6d69351d3c8f3b6`. Its MIT license is included in
`ARCHIFY_LICENSE.txt`. The application does not require Archify to be installed
or contact an external diagram service.
