
<head>

<h1>EEG Classifier</h1>

<h4>Explaining the experiment</h4>
        <div class="section">
            <h4>The participants</h4>
            <p>
                Participants were 61 children with ADHD and 60 healthy controls (boys and girls, ages 7-12).
                The ADHD children were diagnosed by an experienced psychiatrist to DSM-IV criteria,
                and have taken Ritalin for up to 6 months. None of the children in the control group had a history of psychiatric disorders,
                epilepsy, or any report of high-risk behaviors.
            </p>
        </div>

  
<h4>Recording tools</h4>
        <div class="section">
            <p class="highlight">
                EEG recording was performed based on 10-20 standard by 19 channels
                (Fz, Cz, Pz, C3, T3, C4, T4, Fp1, Fp2, F3, F4, F7, F8, P3, P4, T5, T6, O1, O2) at 128 Hz sampling frequency.
                The A1 and A2 electrodes were the references located on earlobes.
            </p>
        </div>


<h4>The Task required</h4>
        <div class="section">
            <p>
                Since one of the deficits in ADHD children is visual attention, the EEG recording protocol was based on visual attention tasks.
                In the task, a set of pictures of cartoon characters was shown to the children, and they were asked to count the characters.
                The number of characters in each image was randomly selected between 5 and 16, and the size of the pictures was large enough to be easily visible
                and countable by children. To have a continuous stimulus during the signal recording, each image was displayed immediately and uninterrupted after the child's response.
                Thus, the duration of EEG recording throughout this cognitive visual task was dependent on the child's performance (i.e. response speed).
            </p>
        </div>


<h4>Refrences</h4>
        <div>
            <p>
                ```bash
                Data is taken from the
                ```
            <a  href="https://ieee-dataport.org/open-access/eeg-data-adhd-control-children"> link </a>
            </p>
        </div>


<h4>CNN Classifier</h4>
        <div class="section">
            <p>
            </p>
        </div>

<h2 style="background-color:blue">Installation required</h2>
<div>
Here's the pip installation command for all required libraries:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

This single command will install all necessary packages:
- numpy: For numerical computations
- pandas: For data manipulation and CSV reading
- scikit-learn: For StandardScaler, train_test_split, and metrics
- tensorflow: Includes keras for deep learning
- matplotlib: For plotting
- seaborn: For enhanced visualizations

Note:
- tensorflow automatically includes keras
- pathlib and os are built-in Python libraries, so they don't need installation
- sys is also a built-in Python library

If you're setting up a new environment, you might want to specify versions for stability:
```bash
pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 tensorflow==2.15.0 matplotlib==3.7.2 seaborn==0.12.2
```
</div>

