# vLLM Prefix Caching 體驗實驗

> 對應投影片:**Chapter 11 / p.76 — A.2 vLLM KV Cache Prefix Hashing**

## 0. 實驗目標

驗證投影片裡兩個量化宣稱:

| 指標 | 投影片宣稱 | 我們要量到 |
| --- | --- | --- |
| 吞吐 | +2~4× | `throughput_req_per_sec`、`throughput_tok_per_sec` |
| 延遲 | -30~50% | `mean TTFT`、`mean E2E latency` |

做法:把同一段 ~800 token 的長前綴(system prompt + 5-shot 範例)接上 100 個不同的後綴問題,**只變動 `enable_prefix_caching=True/False` 兩種設定**,其它都固定,然後比較。

## 1. 為什麼這樣設計就會看到差距?

- 共用前綴有 **~800 tokens**,而 vLLM 預設 block size = 16,所以 ~50 個 block 全部可以被 prefix hash 命中。
- `enable_prefix_caching=True` 時,只有第 1 個 prompt 真的算了前綴 forward,後面 99 個 prompt 命中既有 KV block,**前綴部分 forward 為零**。
- `enable_prefix_caching=False` 時,100 個 prompt 各自重新算 800 token 的 prefix forward,純浪費。
- `SamplingParams(temperature=0, ignore_eos=True, max_tokens=64)` 強制兩次跑出同樣的 token 數量,確保「比的是 cache,不是別的」。

## 2. 環境需求

vLLM **官方不支援原生 Windows**,請從下面三條路擇一:

### 路線 A:Google Colab(最快,有免費 T4)

開新 notebook,跑這幾行:

```python
!pip install -q vllm matplotlib
!git clone https://gist.github.com/yourname/this-repo.git ex || true
%cd ex
!bash run_compare.sh
from IPython.display import Image; Image("comparison.png")
```

> 或者把 `experiment.py` / `plot_results.py` / `run_compare.sh` 直接上傳到 Colab 也可以。

### 路線 B:WSL2 + Ubuntu(本機有 NVIDIA GPU)

```powershell
# 在 PowerShell
wsl --install -d Ubuntu      # 已裝過可跳過
wsl
```

進到 WSL 後:

```bash
sudo apt update && sudo apt install -y python3-pip
pip install -r requirements.txt
cd alg_ch11_p76_ex
bash run_compare.sh
```

或者直接從 Windows PowerShell 一鍵跑:

```powershell
.\run_compare.ps1
```

### 路線 C:純 Linux 主機

```bash
pip install -r requirements.txt
bash run_compare.sh
```

## 3. 一鍵跑

```bash
bash run_compare.sh
```

裡頭做三件事:

1. `python experiment.py --mode off --out result_off.json`
2. `python experiment.py --mode on  --out result_on.json`
3. `python plot_results.py` → 產出 `comparison.png` + 印出加速比

## 4. 預期結果(以 Qwen2.5-0.5B-Instruct + 100 prompts + max_tokens=64 為例)

終端機尾端會看到類似:

```
========== 比較摘要 ==========
  吞吐 (req/s) 加速:  2.7 倍
  吞吐 (tok/s) 加速:  3.1 倍
  平均 TTFT 下降:     38.4 %
  平均 E2E latency 下降: 41.2 %
```

實際數字會隨 GPU 型號、模型大小、prompt 長度浮動,但**方向**應該很穩定:
吞吐倍增、TTFT 與 E2E latency 雙雙下降。

## 5. 同學可以再玩什麼?

把 `experiment.py` 開出來改下面這些參數,觀察差距如何變化:

| 變因 | 實驗想法 | 預期 |
| --- | --- | --- |
| 縮短 `SHARED_PREFIX` 到 50 tokens | prefix < block_size,命中率近 0 | 兩邊差距幾乎消失 |
| 拉長到 1500 tokens | 命中的 block 變多 | 加速比進一步擴大 |
| `--n-prompts 10` vs `1000` | 攤提第 1 次 forward 的成本 | n 越大、加速比越接近極限 |
| `--max-tokens 256` | decode 時間佔比變高 | 加速比下降(cache 只省 prefill) |
| 換成大模型 (e.g. `Qwen2.5-7B-Instruct`) | prefill 計算量更貴 | 加速比通常變更明顯 |

## 6. 檔案清單

```
alg_ch11_p76_ex/
├── README.md          ← 本檔
├── requirements.txt
├── experiment.py      ← 主腳本(跑單一 mode)
├── plot_results.py    ← 讀 JSON 畫圖、印加速比
├── run_compare.sh     ← Linux/WSL/Colab 一鍵跑
└── run_compare.ps1    ← Windows → WSL 一鍵跑
```

## 7. 參考

- Kwon et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention**, SOSP 2023
- vLLM 原始碼:<https://github.com/vllm-project/vllm>
- vLLM Prefix Caching 文件:<https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>

# ----------------------------------------------------

以下為使用colab 詳細操作步驟：

**你要上傳的檔案**

從這個資料夾：

`.\alg_ch11_p76_ex`

上傳這 5 個檔案到 Colab：

```text
experiment.py
plot_results.py
run_compare.sh
requirements.txt
README.md
```

**Colab 手把手步驟**

1. 打開 Google Colab  
   進入：[https://colab.research.google.com](https://colab.research.google.com)

2. 建立新 Notebook  
   點「新增筆記本」或「New notebook」。

3. 開啟 GPU  
   上方選單點：

   `執行階段` → `變更執行階段類型`

   然後設定：

   ```text
   硬體加速器：GPU
   GPU 類型：T4 如果有就選 T4
   ```

   按「儲存」。

4. 連線執行環境  
   右上角按「連線」。  
   等它變成綠色勾勾或顯示 RAM / 磁碟用量。

5. 上傳檔案  
   左邊側欄點「資料夾圖示」，也就是 Files。  
   按「上傳」按鈕，把上面 5 個檔案傳上去。

   上傳後你應該會看到類似：

   ```text
   /content/experiment.py
   /content/plot_results.py
   /content/run_compare.sh
   /content/requirements.txt
   ```

6. 新增第一個程式區塊，安裝套件  
   在 Colab 的程式格輸入：

   ```python
   !pip install -q -r requirements.txt
   ```

   然後按左邊播放鍵執行。  
   這一步可能會跑幾分鐘。

7. 新增第二個程式區塊，確認 GPU 有開到  
   輸入：

   ```python
   !nvidia-smi
   ```

   如果看到 NVIDIA T4 或其他 GPU 資訊，就代表成功。

8. 新增第三個程式區塊，執行實驗  
   輸入：

   ```python
   !bash run_compare.sh
   ```

   這會做三件事：

   ```text
   先跑 prefix caching = off
   再跑 prefix caching = on
   最後產生 comparison.png
   ```

9. 新增第四個程式區塊，顯示圖片結果  
   輸入：

   ```python
   from IPython.display import Image
   Image("comparison.png")
   ```

10. 看文字結果  
   第三個區塊跑完後，最下面應該會看到類似：

   ```text
   ========== 比較摘要 ==========
     吞吐 (req/s) 加速:  2.x 倍
     吞吐 (tok/s) 加速:  3.x 倍
     平均 TTFT 下降:     xx.x %
     平均 E2E latency 下降: xx.x %
   ```

**如果跑太久**

可以先用小一點的測試版本：

```python
!N=10 MT=32 bash run_compare.sh
```

確認流程成功後，再跑正式版：

```python
!bash run_compare.sh
```

**常見狀況**

如果出現沒有 GPU、CUDA、或 vLLM 相關錯誤，先檢查第 3 步有沒有選 GPU。  
如果 Colab 提示要 restart runtime，就重新啟動後，從「安裝套件」那格開始再跑一次。