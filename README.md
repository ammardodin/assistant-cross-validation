# Conversation-Cross-Validation
`Python` utility to cross-validate a conversation workspace

# Instructions
1. Clone the repo.
2. Install [pip](https://pip.pypa.io/en/stable/installing/).
3. Install [Python](http://docs.python-guide.org/en/latest/starting/installation/) if not already available on your system.
4. Install the script's dependencies by running the following command:
```
pip install -r requirements.txt
```
5. Export your conversation instance username and password as environment variables via the following commands:
```
export CONVERSATION_USERNAME=<your conversation service username>
export CONVERSATION_PASSWORD=<your conversation service password>
```
5. Run the script via the command below. The `--folds` argument is optional, but if provided, must be at least `2`.
```
python --data <path to your data> --folds <number of folds>
```