# 🚀 Beginner's Guide: Running Parallax on Kaggle (Cloud Execution)

This guide takes you step-by-step from zero to a completed submission artifact (`matching_results.tsv`) running on Kaggle's free cloud servers with **30 GB of RAM**.

You do **not** need any prior Kaggle experience. Follow these simple steps.

---

## Why Kaggle?
- **30 GB of Cloud RAM** (vs ~14 GB on local machine).
- Runs **independently in the cloud**: you can start the run, turn off your Wi-Fi, shut your laptop lid, and go to sleep.
- Direct high-speed download and upload to **AWS S3** (`s3://amazon-ml-challange-2026-parallax`).

---

## Step 1: Create or Sign In to Your Free Kaggle Account
1. Open your browser and go to: **[https://www.kaggle.com](https://www.kaggle.com)**
2. Click **"Register"** (or **"Sign In"**) in the top right corner. You can sign in instantly with your Google account.
3. **Verify Phone Number (Required once for Internet Access)**:
   - Click your profile icon in the top right $\to$ **Settings**.
   - Under **"Phone verification"**, verify your mobile number.
   *(This is a free one-time verification required by Kaggle to enable notebook internet access).*

---

## Step 2: Open a New Notebook
1. On the left navigation bar, click the **"+ Create"** button (or go to **[https://www.kaggle.com/code](https://www.kaggle.com/code)** and click **"New Notebook"**).
2. A fresh, blank Jupyter notebook will open in your browser.

---

## Step 3: Turn ON "Internet" in Notebook Settings (Crucial Step!)
Kaggle notebooks start with internet disabled by default. You **must enable it** so it can download packages and sync data with AWS S3:

1. Look at the **right-hand sidebar** titled **"Notebook options"** (or **"Settings"**).
   *(If the sidebar is hidden, click the small `|◀` or arrow icon at the top right to expand it).*
2. Scroll down until you see the **"Internet"** option.
3. Switch the toggle to **ON** (it will turn blue/active).
4. *(Optional check)*: Under **"Accelerator"**, leave it as **None / CPU** (or CPU with 4 cores / 30 GB RAM). Our optimized TF-IDF and LightGBM models run blazing fast on CPU.

---

## Step 4: Upload the Parallax Notebook
You have the complete, self-contained notebook already prepared on your computer:

1. In the Kaggle notebook's top menu bar, click **File** $\to$ **Upload Notebook**.
2. Click **Browse** and navigate to:
   `/home/akshit/Projects/hackathon/Parallax/notebooks/parallax_kaggle_runner.ipynb`
3. Click **Open / Upload**.
4. The notebook will load into Kaggle with 5 pre-built cells:
   - **Cell 1**: Installs dependencies (`rapidfuzz`, `jellyfish`, `lightgbm`, `boto3`, etc.).
   - **Cell 2**: Unpacks the Parallax engine and configures AWS S3 credentials.
   - **Cell 3**: Runs the master pipeline (`scripts/run_full_submission.py`).
   - **Cell 4**: Copies and validates `matching_results.tsv` in the Kaggle output directory.

*(Alternative: If you prefer copy-pasting code into a single cell, you can copy the contents of `/home/akshit/Projects/hackathon/Parallax/notebooks/kaggle_script.py` into a blank cell and run it).*

---

## Step 5: Start the Background Run & Close Your Laptop!
You do **not** need to keep your browser tab open or stay connected to Wi-Fi.

1. Click the blue **"Save Version"** button in the top right corner of the Kaggle interface.
2. In the modal that appears:
   - Set **Version type** to: **"Save & Run All (Commit)"**
   - Click **"Save"**.
3. A small status notification in the bottom left will show: *"Creating version... Running"*.
4. **That's it!** Kaggle is now executing the entire pipeline on a cloud server.
5. **You can safely close the browser, turn off Wi-Fi, put your laptop to sleep, or power it down.**

---

## Step 6: Collecting Your Submission in the Morning
The run will complete in approximately **35–45 minutes**.

1. Open **[https://www.kaggle.com/code](https://www.kaggle.com/code)** and click on your notebook.
2. Under the **"Versions"** tab or in the right-hand panel under **"Output"**:
   - You will see **`matching_results.tsv`** (~100 MB).
   - Click the three dots `⋮` next to it and select **"Download"**.
3. Additionally, the notebook **automatically uploads the submission to your AWS S3 bucket**:
   - `s3://amazon-ml-challange-2026-parallax/submissions/matching_results.tsv`
   *(You can also download it from S3 anytime).*

---

## What the Pipeline Executes in the Cloud:
| Stage | Description | Memory Footprint |
| :--- | :--- | :--- |
| **Stage 0-1** | Pulls 100% full raw train & test TSVs directly from S3 | Disk stream |
| **Stage 2-3** | Mines hard negatives across **100% of all 10.3M target records** | Peak RAM ~3.8 GB |
| **Stage 4** | Trains Two-Pass LightGBM GBDT with 54 features | Fast multi-threaded CPU |
| **Stage 5** | Calibrates optimal decision threshold $\tau$ on holdout queries | RAM < 1 GB |
| **Stage 6** | Streams test inference over **all 1,732,544 queries** (India, US, France) | Peak RAM < 4.5 GB |
| **Stage 7** | Enforces official Amazon ML Challenge validator checks | 0 error verification |
| **Stage 8** | Uploads final verified artifact to AWS S3 & writes Kaggle output | ~100 MB artifact |
