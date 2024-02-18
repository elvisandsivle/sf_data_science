CREATE table transactions (
	transaction_id int4 primary KEY
	,product_id int4
	,customer_id int4
	,transaction_date date
	,online_order Bool
	,order_status varchar(30)
	,brand varchar(30)
	,product_line varchar(30)
	,product_class varchar(30)
	,product_size varchar(30)
	,list_price	float4
	,standard_cost float4
)

CREATE table customers (
	customer_id int4 primary KEY
	,first_name varchar(50)
	,last_name varchar(50)
	,gender varchar(30)
	,DOB varchar(50)
	,job_title varchar(50)
	,job_industry_category varchar(50)
	,wealth_segment varchar(50)
	,deceased_indicator varchar(50)
	,owns_car varchar(30) 
	,address varchar(50)
	,postcode varchar(30)
	,state varchar(30)
	,country varchar(30)
	,property_valuation int4
)

select distinct brand
from transactions t
where standard_cost > 1500

select transaction_id, transaction_date, order_status
from transactions t
where order_status = 'Approved'
	and transaction_date
	between '2017-04-01'::date and '2017-04-09'::date
order by transaction_date

select job_title as hw_2_3, job_industry_category
from customers t
where (job_industry_category = 'IT' 
	or job_industry_category = 'Financial Services')
	and job_title like 'Senior%'

select distinct brand
from transactions t
where order_status = 'Approved'
	and customer_id in 
	(select customer_id 
	from customers 
	where job_industry_category = 'Financial Services')

select customer_id, first_name, last_name
from customers
where customer_id in 
	(select customer_id 
	from transactions t 
	where (online_order = True) 
	and (order_status = 'Approved') 
	and (brand = 'Giant Bicycles' or brand = 'Norco Bicycles' or brand = 'Trek Bicycles'))
limit 10

select cust.customer_id, cust.first_name, cust.last_name, trans.transaction_id 
from customers cust
left join transactions trans 
on cust.customer_id = trans.customer_id 
where trans.customer_id is null

select cust.customer_id, cust.first_name, cust.last_name, max(trans.standard_cost)
from customers cust
join transactions trans
on cust.customer_id = trans.customer_id
where cust.job_industry_category = 'IT'
group by cust.customer_id
order by max(trans.standard_cost) desc

select cust.customer_id, cust.first_name, cust.last_name, trans.transaction_date, trans.order_status, cust.job_industry_category
from customers cust
join transactions trans
on cust.customer_id = trans.customer_id
where (cust.job_industry_category = 'IT'
	or cust.job_industry_category = 'Health')
	and trans.order_status = 'Approved'
	and trans.transaction_date
	between '2017-07-07'::date and '2017-07-17'::date
group by cust.customer_id, trans.transaction_date, trans.order_status, cust.job_industry_category
order by trans.transaction_date desc