# Conversation-Cross-Validation

`Python` utility to cross-validate a conversation workspace

# Instructions

1. Install [Python](http://docs.python-guide.org/en/latest/starting/installation/) if not already available on your system.
2. Install [pip](https://pip.pypa.io/en/stable/installing/).
3. Install virtualenv via:

```
pip install virtualenv
```

4. Create a virtual environment and activate it via:

```
virtualenv <name of virtual environment>
source activate <name of virtual environment>/bin/activate
```

5. Clone the repo and change your current working directory to that of the cloned repo.
6. Place your exported workspace csv data in the cloned repo directory.
7. Install the script's dependencies by running the following command:

```
pip install -r requirements.txt
```

8. Export your conversation instance username and password as environment variables via the following commands:

```
export CONVERSATION_USERNAME=<your conversation service username>
export CONVERSATION_PASSWORD=<your conversation service password>
```

9. Run the script via the command below. The `--folds` argument is optional, but if provided, must be at least `2`.

```
python --data <path to your data> --folds <number of folds>
```
