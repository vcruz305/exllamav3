def assigned_positions(start,length,assigned,table):
 assert start>=0 and length>0
 assert start+length<=256*len(assigned), (start,length,assigned)
 assert len(table)>=len(assigned) and table[:len(assigned)]==assigned
 return [(assigned[i//256],i%256) for i in range(start,start+length)]
