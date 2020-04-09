# Solve the wage bill expectations bug

1. **Add "par" attribute to the Agent class**
2. **Correct the "set_Wfs" function so that the wage bill are not set to zero anymore**
3. **Correct the "default_firms_pay_households" function**
    + change name to firms_pay_households
    + the first input will be for all firms not only default firms
    + all firms compute the "par" attribute (how much they can pay to employees)
    + you need a third input "default_fs" in order to identify default firms and 
    the households that loose their jobs
4. **Correct update_Ah and update_Af**
    + multiply wage (bills) with the par attribute
    of the corresponding agent.
    
Those four tasks should solve the problem of systematically downwards adjusted prices.