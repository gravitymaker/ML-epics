epicsEnvSet(P, "ml-nn:")
epicsEnvSet(R, "ex3:")

dbLoadRecords("forecast.db", "P=$(P), R=$(R)")

iocInit
