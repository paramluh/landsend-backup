# Cluster -> node index -> IP, shared by peek.sh, sync.sh and run_multinode.sh.

declare -A NODES_RESERVED_NEBIUS=(
    [0]=192.168.0.4
    [1]=192.168.0.9
    [2]=192.168.0.7
    [3]=192.168.0.8
    [4]=192.168.0.6
)

declare -A NODES_PARAM_NEBIUS=(
    [1]=192.168.0.45
    [2]=192.168.1.7
    [3]=192.168.1.11
    [4]=192.168.0.143
    [7]=192.168.0.14
)

# landsend: names resolve only inside the cluster; reach them via landsend-5's public IP.
LANDSEND_JUMP=vansh@91.194.200.20
declare -A NODES_LANDSEND=(
    [0]=landsend-0
    [1]=landsend-1
    [2]=landsend-2
    [3]=landsend-3
    [4]=landsend-4
    [5]=landsend-5
    [6]=landsend-6
    [7]=landsend-7
    [8]=landsend-8
    [10]=landsend-10
)

declare -A NODES_RICHMOND=(
    [7]=91.239.86.197
    [8]=91.239.86.113
    [9]=91.239.86.198
    [10]=91.239.86.200
    [11]=91.239.86.186
    [12]=91.239.86.20
    [13]=91.239.86.145
    [14]=91.239.86.162
    [15]=91.239.86.139
    [16]=91.239.86.193
)

CLUSTERS=(RESERVED_NEBIUS PARAM_NEBIUS LANDSEND RICHMOND)

# Single-letter cluster convention: r=reserved, p=param, l=landsend, m=richmond.
resolve_cluster() {
    case "${1,,}" in
        r) echo RESERVED_NEBIUS ;;
        p) echo PARAM_NEBIUS ;;
        l) echo LANDSEND ;;
        m) echo RICHMOND ;;
        *) echo "unknown cluster '$1' (r=reserved, p=param, l=landsend, m=richmond)" >&2; return 1 ;;
    esac
}
