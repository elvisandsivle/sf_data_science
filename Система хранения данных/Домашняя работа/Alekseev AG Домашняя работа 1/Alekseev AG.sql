CREATE table transaction (
	transaction_id integer primary KEY
	,product_id integer
	,product_NAME text
	,customer_id integer
	,address_id integer
	,transaction_date timestamp
	,online_order bool
	,order_status bool
	,standart_cost float4
	,list_price float4
)

CREATE table product (
	product_NAME text primary key
    ,product_id integer
    ,brand varchar(50)
    ,product_line varchar(50)
    ,product_class varchar(50)
    ,product_size varchar(50)
    ,standart_cost float4
    ,list_price float4
)

CREATE table customer (
    customer_id integer primary key
    ,first_name varchar(50)
    ,last_name varchar(50)
    ,gender varchar(50)
    ,DOB timestamp
    ,job_title varchar(50)
    ,job_industry_category varchar(50)
    ,wealth_segment varchar(50)
    ,deceased_indicator varchar(50)
    ,owns_car varchar(50)
)

CREATE table address (
    address_id integer primary key
    ,address varchar(50)
    ,postcode integer
    ,state varchar(50)
    ,country varchar(50)
    ,property_valuation integer
)