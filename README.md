# qt3qutipsandbox
Simple examples in qutip and of qutip integration with other qt3 software tools.

## Installation
These instructions assume you've installed [anaconda](https://www.anaconda.com/).  I also recommend the [pycharm community](https://www.jetbrains.com/pycharm/download) IDE for editing and debugging python code.  The instructions also assume that you know how to use the command line "cd" command to change directory-- if you dont know then google around.

Open a terminal (git bash if using windows, terminal on mac or linux). Navigate to the parent folder where you store your git repositories using the 'cd' command in the terminal.
Once there clone the repository and cd into it.
```
git clone https://github.com/qt3uw/qt3qutipsandbox.git
cd qt3qutipsandbox
```
You can use the .yml file contained in the repository to set up an anaconda environment with the required packages using the following command (if you are in windows, you will need to switch from the git bash terminal to the "anaconda prompt" terminal that can be found in start menu if you've installed anaconda, on a mac or linux you can use the good old terminal for everything):
```
conda env create -f environment.yml
```
This creates an anaconda environment called "qt3qutipsandbox", which contains all of the dependencies of this repository.  You can activate that environment with the following command:
```
conda activate qt3qutipsandbox
```
Once activated your terminal will usually show (qt3qutipsandbox) to the left of the command prompt.

Now that your terminal has activated the anaconda environment you can use pip to install this package in that environment.  cd into the parent folder that contains the qt3qutipsandbox repository you cloned earlier.  Then use the following command to install the repository into the qt3qutipsandbox conda environment.
```
pip install -e qt3qutipsandbox
```

### Configure Interpreter in IDE
At this point is complete.  If using an IDE, like pycharm, you will need to make sure that the python interpreter for your project is set to the python.exe file for the anaconda environment that you just created.  An easy way to find the path to that python executable within the environment is to use the following command in a terminal where the qt3qutipsandbox enviornment is activated:
```angular2html
where python
```
On my windows machine in an anaconda prompt this command returns the following (the command itself is the top line):
```
(qt3qutipsandbox) C:\Users\mfpars\repos>where python
C:\Users\mfpars\anaconda3\envs\qt3qutipsandbox\python.exe
C:\Users\mfpars\AppData\Local\Microsoft\WindowsApps\python.exe
```
The path that we want is the middle line (you can see it is showing an anaconda environments folder).  If using pycharm, follow [these instructions](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html#view_list) to set your interpereter to that path.

###

## Examples overview
We have a [presentation](https://docs.google.com/presentation/d/1-dWg_877A0LzcaBWMp4XA1AxHjMvfoKu/edit?usp=sharing&ouid=102194684503910859904&rtpof=true&sd=true) that describes example models and plots some results.


# LICENSE

[LICENSE](LICENSE)
