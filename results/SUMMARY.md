# www_v1 summary

cross runs: 480, indomain runs: 420

## cross mean (HR@20/NDCG@20, avg over directions)
- fcf: 0.2115/0.0778 (n=30)
- scratch: 0.2629/0.0979 (n=30)
- fedprox: 0.2616/0.0984 (n=30)
- fedadam: 0.2562/0.0947 (n=30)
- transfr: 0.2652/0.0986 (n=30)
- l2sp: 0.2593/0.0964 (n=30)
- fedper: 0.2705/0.1019 (n=30)
- pfedme: 0.2749/0.1033 (n=30)
- fedmoe: 0.2731/0.1027 (n=30)
- fedpasa: 0.3053/0.1222 (n=30)
- pasa_no_routing: 0.2625/0.0976 (n=30)
- pasa_no_condition: 0.3029/0.1201 (n=30)
- pasa_no_anchor: 0.3064/0.1226 (n=30)
- pasa_const_routing: 0.2634/0.0985 (n=30)
- pasa_rand_routing: 0.2608/0.0971 (n=30)
- pasa_item_anchor: 0.3063/0.1225 (n=30)

## indomain mean
- fcf: 0.0571/0.0225
- scratch: 0.0691/0.0398
- fedprox: 0.0752/0.0426
- fedadam: 0.0679/0.0389
- fedper: 0.0699/0.0404
- pfedme: 0.0688/0.0401
- fedmoe: 0.0702/0.0406
- fedpasa: 0.0894/0.0384

Wilcoxon fedpasa>transfr HR@20: p=0.0000 (n=30)

Wilcoxon fedpasa>transfr NDCG@20: p=0.0000 (n=30)

FedPASA vs TransFR cross HR gain: 21.6%