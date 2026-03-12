# 分组相对策略优化（GRPO）

在强化学习中，PPO 等经典算法依赖**价值模型（Critic）**来估计动作价值，从而指导学习过程。然而，训练该价值模型会消耗大量计算资源。

GRPO 省去了单独的价值模型，从而简化了训练流程，其工作方式如下：
- **分组采样**：针对给定问题，模型生成多个可行解，形成一组输出。
- **奖励分配**：对每个解按正确性或质量进行评估并分配奖励。
- **基线计算**：以该组的平均奖励作为**基线**。
- **策略更新**：将每个解的奖励与组内基线对比，以此更新模型参数，强化优于平均水平的解，抑制劣于平均水平的解。

该方法无需训练独立的价值估计模型，降低了计算开销，提升了学习效率。更多细节可参考原论文
[DeepSeekMath: 推动开源语言模型数学推理能力极限](https://arxiv.org/pdf/2402.03300)

---

## 核心组件

- **无价值函数（无 Critic）**：与 PPO 不同，GRPO 不训练独立的价值网络（Critic）。
- **分组采样（分组轨迹采样）**：GRPO 不为每个输入只采样一条轨迹，而是对每个提示从当前策略生成多个补全结果（响应），这组结果称为**一个分组**。
- **相对奖励**：在每个分组内对补全结果打分（如基于正确性），并以**组内相对方式**归一化奖励。

---

## 配置说明

注意：所有包含 `micro_batch_size` 的配置均用于设置每次前向/反向传播的最大样本数或 Token 数，以避免 GPU 显存溢出（OOM），其取值**不应改变算法本身或收敛行为**。

尽管许多配置以 `ppo_` 为前缀，但它们在 VERL 框架中可适用于多种强化学习算法，因为 GRPO 的训练流程与 PPO 高度相似（仅不含 Critic）。

- `actor_rollout.ref.rollout.n`：对每个提示采样 n 次。默认为 1。
  使用 GRPO 时请设为**大于 1**，以启用分组采样。

- `data.train_batch_size`：用于生成一组采样轨迹/ rollout 的**全局提示批次大小**。
  最终响应/轨迹总数为：
  `data.train_batch_size * actor_rollout.ref.rollout.n`

- `actor_rollout_ref.actor.ppo_mini_batch_size`：将采样轨迹划分为多个小批次，
  批次大小为 `ppo_mini_batch_size`，用于 PPO 策略网络更新。
  该值为所有工作节点的**全局大小**。

- `actor_rollout_ref.actor.ppo_epochs`：对一组采样轨迹执行 GRPO 更新的轮数。

- `actor_rollout_ref.actor.clip_ratio`：GRPO 截断范围。默认为 0.2。

- `algorithm.adv_estimator`：默认为 gae。使用 GRPO 时请改为 **grpo**。

- `actor_rollout_ref.actor.loss_agg_mode`：默认为 `"token-mean"`。
  可选：`"token-mean"`、`"seq-mean-token-sum"`、`"seq-mean-token-mean"`。
  原始 GRPO 论文采用样本级损失（`seq-mean-token-mean`），在长思维链（CoT）场景下可能不稳定。
  VERL 提供的所有 GRPO 示例脚本均使用默认配置 `"token-mean"` 进行损失聚合。

GRPO 不在奖励中加入 KL 惩罚，而是**直接在损失中加入**当前策略与参考策略之间的 KL 散度来实现正则化：

- `actor_rollout_ref.actor.use_kl_loss`：是否在策略网络中使用 KL 损失。
  启用后不再在奖励函数中加入 KL。默认为 False。
  使用 GRPO 时请设为 **True**。

- `actor_rollout_ref.actor.kl_loss_coef`：KL 损失系数。默认为 0.001。

- `actor_rollout_ref.actor.kl_loss_type`：支持 `kl(k1)`、`abs`、`mse(k2)`、`low_var_kl(k3)`、`full`。
  末尾添加 `+`（如 `k1+`、`k3+`）会使用直通估计，以 k2 获得无偏梯度，
  与 KL 值估计方式无关（详见：https://github.com/volcengine/verl/pull/2953#issuecomment-3162113848）。
  关于策略与参考策略之间 KL 散度的计算方式，可参考：http://joschu.net/blog/kl-approx.html

---

## 高级扩展

### DrGRPO

论文 [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/pdf/2503.20783) 指出：
GRPO 存在**优化偏置**，会导致生成的响应人为变长，尤其在错误输出上更明显。
这种低效性源于 GRPO 使用**分组奖励归一化**计算优势函数，可能无意中偏好更长但精度更低的响应。

DrGRPO 改用**全局常数归一化**来聚合 Token 级损失，从而**消除长度偏置**。

启用 DrGRPO 只需修改以下配置，其余参数与 GRPO 一致：

- `actor_rollout_ref.actor.loss_agg_mode`：设为 `"seq-mean-token-sum-norm"`，关闭序列维度平均。
- `actor_rollout_ref.actor.loss_scale_factor`：（可选）设为一个常数整数（如最大响应长度），
  保证训练过程中归一化尺度一致。未设置时使用当前批次的响应长度。
- `actor_rollout_ref.actor.use_kl_loss`：DrGRPO 请设为 **False**。
- `algorithm.norm_adv_by_std_in_grpo`：设为 **False**，关闭标准差归一化。

---

## 参考示例

Qwen2.5 GRPO 训练日志与命令：[链接](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b-fsdp2.log)

```bash
bash examples/grpo_trainer/run_qwen3-8b.sh
```

更多性能参考请见：https://verl.readthedocs.io/en/latest/algo/baseline.html