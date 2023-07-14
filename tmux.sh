# activate tmux; setup
tmux new-session -d -s 'sess1'
tmux send-keys -t sess1 ". ./venv2/bin/activate" Enter
tmux send-keys -t sess1 "cd sc_project" Enter # cd training
tmux send-keys -t sess1 "nvidia-smi" Enter
# tmux send-keys -t sess1 "echo 'ready'" Enter
tmux split-window -h -t sess1
tmux send-keys -t sess1 ". ./venv2/bin/activate" Enter
tmux send-keys -t sess1 "cd sc_project" Enter # cd training
tmux attach -t sess1