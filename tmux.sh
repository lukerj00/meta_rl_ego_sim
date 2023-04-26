# activate tmux; setup
tmux new-session -d -s 'sess1'
tmux send-keys -t sess1 ". ./venv3/bin/activate" Enter
tmux send-keys -t sess1 "cd training" Enter
tmux send-keys -t sess1 "nvidia-smi" Enter
# tmux send-keys -t sess1 "echo 'ready'" Enter
tmux split-window -h -t sess1
tmux send-keys -t sess1 ". ./venv3/bin/activate" Enter
tmux send-keys -t sess1 "cd training" Enter
tmux attach -t sess1