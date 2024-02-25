SELECT COUNT(customer_id), job_industry_category
FROM Customers
GROUP BY job_industry_category
order by COUNT(customer_id) desc

select SUM(trans.list_price) AS sum, EXTRACT(MONTH from trans.transaction_date) AS month, cust.job_industry_category
FROM Transactions trans
join customers cust
on cust.customer_id = trans.customer_id
group BY month, cust.job_industry_category
order by month, cust.job_industry_category

select count(trans.online_order), trans.online_order, trans.brand, trans.order_status, cust.job_industry_category
FROM Transactions trans
join customers cust
on cust.customer_id = trans.customer_id
where trans.online_order = true
	and trans.order_status = 'Approved'
	and (trans.brand is not null) and (LENGTH(trans.brand) > 0)
	and cust.job_industry_category = 'IT'
group by trans.brand, trans.order_status, cust.job_industry_category, trans.online_order

select sum(list_price), count(list_price), max(list_price), min(list_price), customer_id
from transactions t 
group by customer_id
order by sum(list_price) desc, count(list_price) desc 

select list_price, customer_id,
sum(list_price) OVER(PARTITION BY customer_id) as "sum",
max(list_price) OVER(PARTITION BY customer_id), 
min(list_price) OVER(PARTITION BY customer_id),
count(list_price) over (PARTITION BY customer_id)
from transactions t
order by count desc, sum desc

SELECT first_name, last_name
FROM customers
where customer_id = (SELECT customer_id
from (select customer_id , SUM(list_price)
	from transactions t
	group by customer_id
	order by SUM desc
	limit 1))
	
SELECT first_name, last_name
FROM customers
where customer_id = (SELECT customer_id
from (select customer_id , SUM(list_price)
	from transactions t
	group by customer_id
	order by SUM
	limit 1))	

Select * 
from (Select transaction_id, transaction_date, customer_id,
      RANK() OVER(partition by customer_id Order by transaction_date) firsttransaction
      From transactions)
Where firsttransaction=1
      
SELECT first_name, last_name, job_title
FROM customers
where customer_id = (
	select customer_id
	from (	select customer_id, transaction_date, 
			lead(transaction_date) OVER(PARTITION BY customer_id order by transaction_date) - transaction_date as delta
			from transactions t
			order by delta desc)
	where delta is not null
	group by customer_id
	order by max(delta) desc
	limit 1)
